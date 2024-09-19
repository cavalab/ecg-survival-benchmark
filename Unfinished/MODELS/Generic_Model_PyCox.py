"""
Generic_Model_PyCox is a Generic_Model modified to better fit PyCox models.

Many Generic_Model functions are overwritten.

Like Generic_Model, this provides generic functions that a specific_pycox_model might want to use

# Additional args expected:
# args['num_durations']  # int, how many discrete time points to use
# args['pycox_mdl']      # str, which survival model to use. one of ['LH', 'MTLR', 'DeepHit','CoxPH']


Does:
- Fixes a bug in DeepHitSingle loss

Samplers and Datasets:   
- Custom_Sampler: a Sampler for a Dataloader that guarantees 1 positive case per batch
- Dataset_FuncList: a Dataset that applies functions, from a list, to each sample, in order. Can return either X or (X,Y) 
    Note:
    - A DataLoader tells a Sampler to provide it with a list of indices. These go to a Dataset, which provides data. The Dataloader manages data batching and re-initializing Samplers when the epoch ends.
    - PyCox handles its own training, so we never get to access a 'batch' of samples. Any changes we want to make to a data (e.g. normalization) have to be front-loaded (which wouldn't work for e.g. augmentation) or done per-sample by the 'Dataset'
    - In retrospect, I probabaly should have had the "Dataset" return a "batch" and have the 'DataLoader' go with a "batch size" of one

- Process_Args_PyCox: Runs Generic_Model.Process_Args(), then processes args generic to PyCox models
- Prep_Data_Normalization_Discretization: prepares normalization AND discretization (3 of 4 PyCox models want 'time' to be an integer bin)
- Process_Data_To_Dataloaders: Here, unlike Generic_Model, we front-load data normalizaiton. also front-loads discretization
- Get_PyCox_Model: Loads the right PyCox model. wraps model.
- Train: PyCox handles its own training, so we run that 1 epoch at a time.
- Run_NN: Evaluates a PyCox model (during validation or testing)
- Save_NN_PyCox: Saves all model components
- Test: wraps Run_NN to return evaluation results

"""


import torch
import numpy as np
import time
import os

from MODELS.GenericModel import GenericModel

from MODELS.Support_Functions import Save_Train_Args

        
import torchtuples as tt
from pycox.models import LogisticHazard
from pycox.models import MTLR
from pycox.models import CoxPH # Bug on loss function if all events 0
from pycox.models import DeepHitSingle

import pandas as pd


import copy

# models
from MODELS.Ribeiro_Classifier import get_ribeiro_model
from MODELS.Ribeiro_Classifier import get_ribeiro_process_multi_image
from MODELS.Ribeiro_Classifier import get_ribeiro_process_single_image

from MODELS.InceptionTimeClassifier import get_InceptionTime_model
from MODELS.InceptionTimeClassifier import get_InceptionTime_process_multi_image
from MODELS.InceptionTimeClassifier import get_InceptionTime_process_single_image

# %% Bugfix - DeepHitSingle, from pcwangustc
# https://github.com/havakv/pycox/issues/79
from pycox.models import loss as pycox_loss
from pycox.models.data import pair_rank_mat

def deephit_loss(scores, labels, censors):
    rank_mat = pair_rank_mat(labels.cpu().numpy(), censors.cpu().numpy())
    rank_mat = torch.from_numpy(rank_mat)
    rank_mat = rank_mat.to('cuda')
    loss_single = pycox_loss.DeepHitSingleLoss(0.2, 0.1)
    loss = loss_single(scores, labels, censors, rank_mat)
    return loss

# %% Datasets, Samplers, and Collate functions
# -- Custom Data Sampler - PyCoxPH needs at least one positive sample per batch
from torch.utils.data.sampler import BatchSampler
class Custom_Sampler(BatchSampler):
    """
    Returns indicies s.t. at least one example of Event=1 per batch.
    ... And does that randomly, without replacement.
    ... Hobbled together from pytorch documentation.
    This is re-created differently every epoch by the DataLoader
    """
    def __init__(self, y_data, batch_size):
        # sampler gets 'data', so PID is in -3, event in -1, time in -2
        # 1. figure out length
        # 2. figure out which indices correspond to '0's (censor) and '1's  (event)
        self.num_samples = len(y_data)
        self.y_data = y_data
        self.batch_size = batch_size
        self.Event_Inds = [i for i,k in enumerate(y_data[:,-1]) if k==1.0]
        self.Non_Event_Inds = [i for i,k in enumerate(y_data[:,-1]) if k==0.0]
        self.weights = torch.tensor([1.0 for k in range(y_data.shape[0])])
        self.default_order = False 
        
    def __iter__(self):
        if (self.default_order == False):
            
            #3. Start by shuffling the dataset
            Random_Indices = torch.multinomial(self.weights, self.num_samples, False)
            Num_Replacements_to_Prep = int(self.num_samples / self.batch_size) + 1
            
            #4. Mark ceil(num_samples / batch_size) event cases that we can insert throughout this dataset to guarantee that each batch gets one positive case
            Replacement_Indices = torch.multinomial(torch.tensor([1.0 for k in self.Event_Inds]), Num_Replacements_to_Prep, True)
            Replacement_Indices = [self.Event_Inds[k] for k in Replacement_Indices]
            
            #5. Parse the shuffled order. If at any point we don't see an event for batch_size indices in a row, replace the last (non-event) index with an event idnex
            Replace_With_Index = 0
            k=0
            while (k < self.num_samples):
                start_ind = k
                end_ind = min(k+self.batch_size, self.num_samples)
                if (sum (self.y_data[Random_Indices[start_ind:end_ind],1]) <0.5):
                    Random_Indices[end_ind-1] = Replacement_Indices[Replace_With_Index]
                    Replace_With_Index = Replace_With_Index+1
                k = k + self.batch_size
       
            yield from iter(Random_Indices)
            
        # sometimes (evaluation) we just want to return the dataset without shuffling
        else:
            yield from range(self.num_samples)

    def __len__(self):
        return self.num_samples
        
#  Dataset functions. PyCox needs everything either frontloaded or done by the dataset (can't access the batch after it's loaded to e.g. normalize the whole thing)
class Dataset_FuncList(torch.utils.data.Dataset):
    # Applies functions in func_list, in order, to x, 
    # Returns x, (y[0], y[1]). y[0] is time (float32), y[1] is event (0 or 1, int)
    # x,y must be Tensor for CollateFunc
    # ... Return_Toggle changes if just x, or x,y are returned
    
    def __init__(self, data, targets = None, covariates = None, func_list= None, discretize = False, Toggle = 'XY'):
        
        self.data = data
        if (targets is not None):
            if (discretize):
                self.targets = torch.tensor(targets.astype(np.int64)) # must be tensor and integer IF DISCRETIZING
            else:
                self.targets = torch.tensor(targets.astype(np.float64)) # must be tensor 
        else:
            self.targets = None
        self.func_list = func_list
        self.Return_Toggle = Toggle # 'X' - return only X. 'XY' - return both in PyCox formats
        
        self.Covariates = torch.tensor(covariates).to(torch.float32) # covariates, like ECG, will be float32

    def __getitem__(self, index): 
        if isinstance(index, slice): # ... Only LH fit requests 2 elements as a slice... to check their size?
            return [1,2,3] # couldn't actually get it to be happy with returning elements. This seems to work just fine
            
        x = self.data[index] # pointer, if modifying, torch.clone first (else permanent)
        if (self.func_list is not None):
            x = torch.clone(x)
            for k in self.func_list:
                x = k(x)
                
        z = self.Covariates[index] # let's include covariates just like this
        # z = torch.tensor(z)
        
        if (self.Return_Toggle =='XY'):
            y = self.targets[index]
            return (x, z), (y[0], y[1].to(torch.float32)) #must be tensor, (int64tensor if discretizing, else float32/64), float32tensor. Outputs as tuple.
        if (self.Return_Toggle =='X'):
            return (x, z) # must be a tensor, NEEDS A COMMA TO BE A TUPLE containing a tensor
    
    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    # batch is a list of tuples of tensors
    # this takes that list, stacks the tensors, and returns a tuple of stacked tensors
    # ultimately we have to match ... input, target = dataloader()
    # ^ lumping x,z into a tuple in the dataset works
    return tt.tuplefy(batch).stack() # demands list of tuples of torch tensors

# %%  ---------------- Start the model
class Generic_Model_PyCox(GenericModel):

    def __init__(self, args, Data):
        
        
        # 0. Process input arguments 
        self.process_args_PyCox(args)
        self.Data = Data
        
        # 1. Adjust data for this model
        self.restructure_data()
        self.prep_normalization_parameters()
        self.prep_data_discretization()
        
        self.normalize_data()
        if (self.max_duration is not None):
            self.discretize_data()

        # 2. Grab ECG processing network 
        self.gen_ecg_model()
        
        # 3. wrap our network with the fusion modules before building optimizer/scheduler
        self.prep_fusion(out_classes = self.Num_Classes)
        
        # 4. Wrap with PyCox model and init optimizer and scheduler
        self.pycox_mdl = self.Get_PyCox_Model() # sets pycox_mdl, optimizer, scheduler
        
        # 5. build dataloaders (requires ECG processing model)
        self.prep_dataloaders()
        

# %% the model
    def gen_ecg_model(self):
        # figures out how to summon model, image adjustment functions
        
        # 1 figure out output channel size
        if 'x_train' in self.Data.keys():
            n_in_channels = self.Data['x_train'].shape[-1]
        else:
            n_in_channels = 12
        
        if (self.args['Model_Type'] == 'Ribeiro'):
            self.model = get_ribeiro_model(self.args, n_in_channels) # get the ECG interpreting model
            self.Adjust_Many_Images = get_ribeiro_process_multi_image() # pointer to function
            self.Adjust_One_Image = get_ribeiro_process_single_image() # pointer to function
            
        if (self.args['Model_Type'] == 'InceptionTime'):
            self.model = get_InceptionTime_model(self.args, n_in_channels) # get the ECG interpreting model
            self.Adjust_Many_Images = get_InceptionTime_process_multi_image() # pointer to function
            self.Adjust_One_Image = get_InceptionTime_process_single_image() # pointer to function

# %% augment process_args, models should include this in init
    def process_args_PyCox(self, args):
        self.Process_Args(args) # call generic_model's arg processing
        
        # now add a few that are generic to pycox models
        if ('num_durations' not in args.keys()):
            args['num_durations'] = '100'
            print('By default, using 100 time intervals')
            self.num_durations = 100
        else:
            self.num_durations = int(args['num_durations'])
        
        # Decide things based on pycox model:
        if ('pycox_mdl' not in args.keys()):
            print('pycox_mdl not in args. exiting')
            quit()
        
        if (self.args['pycox_mdl'] in ['LH', 'MTLR', 'DeepHit']): # are we discretizing data before training/running?
            self.Num_Classes = self.num_durations
            self.Discretize = True
        else:
            self.Num_Classes = 1
            self.Discretize = False

# %% data discretization
    def prep_data_discretization(self):
        self.max_duration = None
        if 'x_train' in self.Data.keys():
            self.max_duration = max(self.Data['y_train'][:,-2]) # time now lives @ -2
            self.labtrans = LogisticHazard.label_transform(self.num_durations)
            self.labtrans.fit_transform(np.array([0,self.max_duration]), np.array([0,1]))            

    def discretize_data(self):
        a = time.time()
        if (self.Discretize): # discretizses time-to-event based on [100] time segments
            for key in ['y_train', 'y_valid', 'y_test']:
                self.Data[key][:,-2], self.Data[key][:,-1] = self.labtrans.transform(self.Data[key][:,-2], self.Data[key][:,-1]) # TTE lives in [:,-2], Event lives in [:,-1]
        print('GenericModelPyCox: discretize_data T = ', '{:.2f}'.format(time.time()-a))

# %% Overwrite Process_Data_To_Dataloaders from Generic_Model
# This runs AFTER we have normalization and time discretization prepped from either init or load
# The goal is to get everything ready for PyCox model training
# 1) Prep which functions get called per image
# 2) Discretize or normalize Data
# 3) Prep Datasets and DataLoaders

    def prep_dataloaders(self):
        a = time.time()
        func_list = []
        func_list.append(self.Adjust_One_Image) # adjust each ecg individually after loading
        
        if 'x_train' in self.Data.keys():
            self.train_dataset = Dataset_FuncList(self.Data['x_train'] , targets = self.Data['y_train'][:,[-2,-1]], covariates = self.Data['z_train'], func_list = func_list, discretize=self.Discretize)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.GPU_minibatch_limit, collate_fn=collate_fn, sampler=Custom_Sampler(self.Data['y_train'], self.GPU_minibatch_limit)) 

        if 'x_valid' in self.Data.keys():
            self.val_dataset  = Dataset_FuncList(self.Data['x_valid']  , targets = self.Data['y_valid'][:,[-2,-1]], covariates = self.Data['z_valid'], func_list = func_list, discretize=self.Discretize )
            self.val_dataloader = torch.utils.data.DataLoader (self.val_dataset,  batch_size = self.GPU_minibatch_limit, collate_fn=collate_fn, sampler=Custom_Sampler(self.Data['y_valid'], self.GPU_minibatch_limit)) 

        if 'x_test' in self.Data.keys():
            self.test_dataset = Dataset_FuncList(self.Data['x_test'], targets = self.Data['y_test'][:,[-2,-1]], covariates = self.Data['z_test'], func_list = func_list, discretize=self.Discretize, Toggle='X') #Only returns X, not Y
            self.test_dataloader = torch.utils.data.DataLoader (self.test_dataset,  batch_size = self.GPU_minibatch_limit, collate_fn=collate_fn, shuffle = False) #DO NOT SHUFFLE
        print('GenericModel_PyCox: Dataloader prep T = ', '{:.2f}'.format(time.time()-a))

# %% Prep pycox model (here so as not to duplicate in train, load, and run)
    def Get_PyCox_Model(self):
        
        self.prep_optimizer_and_scheduler()
        
        # note: pycox_mdl optimizer is in pycox_mdl.optimizer.optimizer
        if (self.args['pycox_mdl'] == 'LH'):
            pycox_mdl = LogisticHazard(self.model, self.optimizer, duration_index=self.labtrans.cuts)  
        if (self.args['pycox_mdl'] == 'MTLR'):
            pycox_mdl = MTLR(self.model, self.optimizer, duration_index=self.labtrans.cuts)
        if (self.args['pycox_mdl'] == 'CoxPH'):
            pycox_mdl = CoxPH(self.model, self.optimizer)
        if (self.args['pycox_mdl'] == 'DeepHit'):
            pycox_mdl = DeepHitSingle(self.model, self.optimizer, duration_index=self.labtrans.cuts, loss = deephit_loss)
            
        return pycox_mdl


# %% Overwrite Train from Generic_Model
    def train(self):
        # store a copy of the best model available
        Best_Model = copy.deepcopy(self.model)

        train_loss = -1 # in case no training occurs
        # Try to load a checkpointed model?
        if self.epoch_end > self.epoch_start:
            print('Generic_Model_PyCox.Train(): Training Requested. Loading best then last checkpoints.')
            last_checkpoint_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
            best_checkpoint_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
            if (os.path.isfile(last_checkpoint_path)):
                if (os.path.isfile(best_checkpoint_path)):
                    self.Load('Best')
                    Best_Model = copy.deepcopy(self.model)
                    print('Generic_Model_PyCox.Train(): Best Checkpoint loaded and Best model copied.')
                    self.Load('Last')
                    print('Generic_Model_PyCox.Train(): Checkpointed model loaded. Will resume training.')
                    
                    val_perfs = np.array([k[2] for k in self.Perf])
                    if (self.early_stop > 0):
                        if (len(val_perfs) - (np.argmin(val_perfs) + 1 ) ) >= self.early_stop:
                            # ^ add one: len_val_perfs is num trained epochs (starts at 1), but argmin starts at 0.
                            print('Generic_Model_PyCox.Train(): Model at early stop. Setting epoch_start to epoch_end to cancel training')
                            self.epoch_start = self.epoch_end
                    
                    if (self.epoch_start == self.epoch_end):
                        print('Generic_Model_PyCox.Train(): Loaded checkpointed model already trained')
                    if (self.epoch_start > self.epoch_end):
                        print('Generic_Model_PyCox.Train(): Requested train epochs > current epochs trained. evaluating.')
                        self.epoch_start = self.epoch_end
                        # breakpoint()
                else:
                    print('Generic_Model_PyCox.Train(): FAILED to load best model! Eval may be compromised')
            else:
                print('Generic_Model_PyCox.Train(): Last checkpoint unavailable.')
                

        # Train
        for epoch in range(self.epoch_start, self.epoch_end):

            epoch_start_time = time.time()
            pycox_log = self.pycox_mdl.fit_dataloader(self.train_dataloader, epochs=1, verbose=True, val_dataloader=self.val_dataloader) 
            epoch_end_time = time.time()
            
            # get train, val loss
            temp = pycox_log.get_measures()
            temp = temp.split(',') 
            train_loss = float(temp[0].split(':')[1])
            val_loss   = float(temp[1].split(':')[1])
            
            # update scheduler # no effect unless args['Scheduler'] == 'True'
            if (hasattr(self,'scheduler')):
                self.scheduler.step(val_loss)
                tmp_LR = self.pycox_mdl.optimizer.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                tmp_LR = 0
            
            # ----
            # Run Validation and Checkpoint
            if ( (epoch+1) % self.validate_every ==0):
        
                # If this is the new best model, save it as the best model
                if val_loss < self.Val_Best_Loss: 
                    nn_file_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
                    if (self.Save_Out_Best):
                        Save_NN_PyCox(epoch, self.model, nn_file_path, optimizer=None, scheduler=None, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev, max_duration=self.max_duration)
                        Best_Model = copy.deepcopy(self.model) # store a local copy of the model
                    Save_Train_Args(os.path.join(self.model_folder_path,'Train_Args.txt'), self.args)
                    self.Val_Best_Loss = val_loss
                    
                # And checkpoint model in any case
                nn_file_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
                if (self.Save_Out_Checkpoint):
                    if (hasattr(self,'scheduler')):
                        Save_NN_PyCox(epoch, self.model, nn_file_path, optimizer = self.optimizer, scheduler=self.scheduler, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev, max_duration=self.max_duration)
                    else:
                        Save_NN_PyCox(epoch, self.model, nn_file_path, optimizer = self.optimizer, scheduler=None, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev, max_duration=self.max_duration)
                Save_Train_Args(os.path.join(self.model_folder_path,'Train_Args.txt'), self.args)
                
                # Update Progress
                new_perf = [epoch, train_loss, val_loss, tmp_LR , epoch_end_time - epoch_start_time]
                print(new_perf)
                self.Perf.append(new_perf)
                
                # Log Progress              
                csv_file_path = os.path.join(self.model_folder_path, 'Training_Progress.csv')
                np.savetxt(csv_file_path, np.asarray(self.Perf), header = "Epoch,Train Loss, Validation Loss, LR, Runtime seconds",delimiter = ',')
                
                # save out performance curves
                Perf_Plot_Path = os.path.join(self.model_folder_path, 'Training_Plot.png')
                self.Save_Perf_Curves(self.Perf, Perf_Plot_Path)
                
                # consider stopping based on early stop
                val_perfs = np.array([k[2] for k in self.Perf])
                if (self.early_stop > 0):
                    if (len(val_perfs) - (np.argmin(val_perfs) + 1 ) ) >= self.early_stop:
                        # ^ add one: len_val_perfs is num trained epochs (starts at 1), but argmin starts at 0.
                        break
        
        self.model = copy.deepcopy(Best_Model)
        self.pycox_mdl.net = self.model
        return train_loss

# %% Model Loading
        
    # After we load a model, we might need to re-do discretization. Currently, this has no effect since we load the training data each time
    # def Discretize_On_Load(self, Import_Dict):
    #     if ('max_duration' in Import_Dict.keys()):
    #         if (self.Num_Classes > 1): # If we're loading an LH/MTLR/DeepHit model, overwrite number of discretization time steps
    #             self.num_durations = self.Num_Classes
    #         self.max_duration = Import_Dict['max_duration']
    #         self.labtrans = LogisticHazard.label_transform(self.num_durations)
    #         self.labtrans.fit_transform(np.array([0,self.max_duration]), np.array([0,1])) 
    #     else:
    #         print('cant discretize data on load')
    #         exit()
            
        # If we _did_ want to process a new test set without loading the training set, we'd need to normalize/discretize the data after loading an existing model:
        # if (self.max_duration is not None):
        #     if ( ('Train' not in self.args.keys()) or (self.args['Train'] == 'False') ):
        #         self.Process_Data_To_Dataloaders()

# %% Overwrite Run, include output discretization from continuous CoxPH model
    def Run_NN (self, my_dataloader):

        if (self.args['pycox_mdl'] == 'CoxPH'):
            # ugly, but necessary

            # okay, so we're passing input,target
            # input needs to be a tuple of tensors
            
            # don't actually store to any variables - can be expensive
            # a = self.Adjust_Many_Images(self.Data['x_train'][:,0,:,:])
            # b = torch.tensor(self.Data['z_train']).to(torch.float32)
            # self.pycox_mdl.compute_baseline_hazards(input= (a,b), target=[self.Data['y_train'][:,-2],self.Data['y_train'][:,-1]],batch_size = self.GPU_minibatch_limit)
            
            self.pycox_mdl.compute_baseline_hazards(input= (self.Adjust_Many_Images(self.Data['x_train']),torch.tensor(self.Data['z_train']).to(torch.float32)), target=[self.Data['y_train'][:,-2],self.Data['y_train'][:,-1]],batch_size = self.GPU_minibatch_limit)
            # note: because we use validation performance to pick the model, we can't build the baselines on validation while that is going on (that's like fitting on the test set)
           
        surv    = self.pycox_mdl.predict_surv(my_dataloader)
        surv_df = self.pycox_mdl.predict_surv_df(my_dataloader) # contains surv, also stores 'index' which is the time in years rather than discretized points

        # for CoxPH you have to discretize time points manually
        if (self.args['pycox_mdl'] == 'CoxPH'):
            # survival (x) is probability that event occurs AFTER the current time point (surv[:,0] != 1)
            # but this is evaluated at every single TTE.
            # ex: 738 TTEs that we want to map to 100 discrete times... but they only fall into 90 of them
            # So: 
            # parse columns of output: [large] x 100
            # parse columns of input: [large] x 738. 
            # input columns align with output columns via 't2': ex: array([ 1,  1,  2,  3,  6, 10, 10]). 90 unique values.
            # where a time maps to a bin, fill that in, if there's a gap, fill in prev known value
            Unique_Time_Points = np.unique(self.Data['y_train'][:,-2])
            t2, k = self.labtrans.transform(Unique_Time_Points, np.ones(Unique_Time_Points.shape))
            surv_out = np.ones( (my_dataloader.dataset.targets.shape[0],len(self.labtrans.cuts)), dtype=float)
            temp_col = surv[:,0]
            for k in range(surv_out.shape[1]):  # parse bins k
                for i,m in enumerate(t2):       # parse TTEs m
                    if (m > k):                 # if m is associated with a later k, look at next bin
                        break
                    if (m==k):                  # if m maps to k, remember which column we looked at
                        temp_col = surv[:,i]    # [this repeats until we get the TTE col associated with bin k]
                surv_out[:,k] = temp_col       # after parsing all input columns m, we have the highest m <= k

            surv = surv_out
            t, d = self.labtrans.transform(my_dataloader.dataset.targets[:,0].numpy(),my_dataloader.dataset.targets[:,1].numpy()) # In 'dataset', time lives at [:,0] and event lives at [:,1]
            surv_df = pd.DataFrame(data = np.transpose(surv), columns = [k for k in range(surv.shape[0])], index = self.labtrans.cuts )

        # if not pycox, you already have t,d discretized in Data[]
        else: 
            t = np.array([int(k) for k in my_dataloader.dataset.targets[:,0]])
            d = np.array([int(k) for k in my_dataloader.dataset.targets[:,1]])
            
            
        cuts = self.labtrans.cuts 

        return cuts, t, d, surv , surv_df

# %% Overwrite test    
    # %% Test
    def Test(self, Which_Dataloader = 'Test'):

        if (Which_Dataloader == 'Train'): # If we want to evaluate on train, we need to change the dataloader return param first
            self.train_dataloader.dataset.Return_Toggle = 'X'
            self.val_dataloader.sampler.default_order = True # and don't shuffle
            cuts, t, d, surv, surv_df  = self.Run_NN(self.train_dataloader) 
            self.val_dataloader.sampler.default_order = False
            self.train_dataloader.dataset.Return_Toggle = 'XY'
        elif (Which_Dataloader == 'Validation'):
            self.val_dataloader.dataset.Return_Toggle = 'X' # only return x
            self.val_dataloader.sampler.default_order = True # and don't shuffle
            cuts, t, d, surv, surv_df  = self.Run_NN(self.val_dataloader) 
            self.val_dataloader.sampler.default_order = False
            self.val_dataloader.dataset.Return_Toggle = 'XY'
        else:
            cuts, t, d, surv, surv_df  = self.Run_NN(self.test_dataloader) # this is an UNSHUFFLED dataloader sent through TorchTuples (only output x, can't recover shufled y if shuffled)
        return cuts, t, d, surv, surv_df

# %% Model Loading
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)   
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        self.Load_Normalization(Import_Dict) # we're frontloading normalization, so that doesn't matter

        self.model.load_state_dict(Import_Dict['model_state_dict'])
        
        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        else:
            print('NO optimizer loaded')
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
        else:
            print("NO scheduler loaded")

        self.Load_Random_State(Import_Dict)


# %% Overwrite save
def Save_NN_PyCox(epoch, model, path, best_performance_measure=9999999, optimizer=None, scheduler=None, NT=None, NM=None, NS=None, max_duration=None):
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    # best_performance_measure refers to the performance of the best model so far
    # so we don't accidentally overwrite it
    
    Out_Dict = {}
    Out_Dict['epoch'] = epoch
    Out_Dict['model_state_dict'] = model.state_dict()
    Out_Dict['Numpy_Random_State'] = np.random.get_state()
    Out_Dict['Torch_Random_State'] = torch.get_rng_state()
    Out_Dict['CUDA_Random_State'] = torch.cuda.get_rng_state()
    Out_Dict['best_performance_measure'] = best_performance_measure
    
    if (optimizer is not None):
        Out_Dict['optimizer_state_dict'] = optimizer.state_dict()
    if (scheduler is not None):
        Out_Dict['scheduler_state_dict'] = scheduler.state_dict()
        
    # Normalization Parameters
    if (NT is not None):
        Out_Dict['NT'] = NT # normalization type
    if (NM is not None):
        Out_Dict['NM'] = NM # normaliation mean per channel
    if (NS is not None):
        Out_Dict['NS'] = NS # normalization stdev per channel
        
    # Time discretization - if you have the number of cuts (which you get from model size) and the max duration, you can recreate the discretization
    if (max_duration is not None):
        Out_Dict['max_duration'] = max_duration
    
    torch.save(Out_Dict, path)
    
