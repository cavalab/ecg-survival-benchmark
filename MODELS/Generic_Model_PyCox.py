# Args expected:
# args['spect']          # str, 'FFT' or 'Welch' queue up those transforms (after normalization and discretization, but before anything else)
# args['num_durations']  # int, how many discrete time points to use
# args['pycox_mdl']      # str, which survival model to use. one of ['LH', 'MTLR', 'DeepHit','CoxPH']

import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from MODELS.Generic_Model import Generic_Model

from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss
        
import torchtuples as tt
import pycox
from pycox.models import LogisticHazard
from pycox.models import MTLR
from pycox.models import CoxPH # Bug on loss function if all events 0
from pycox.models import CoxCC # no Idea how to set up dataloader
from pycox.models import DeepHitSingle

import pandas as pd
from pycox.evaluation import EvalSurv

from numpy.fft import fft
from scipy.signal import welch

import copy

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

# %% Custom Data Sampler - PyCoxPH needs at least one positive sample per batch
from torch.utils.data.sampler import BatchSampler
class Custom_Sampler(BatchSampler):
    """
    Returns indicies s.t. at least one example of Event=1 per batch.
    ... And does that randomly, without replacement.
    ... Hobbled together from pytorch documentation.
    This is re-created every epoch by the DataLoader
    """
    def __init__(self, y_data, batch_size):
        # 1. figure out length
        # 2. figure out which indices correspond to '0's (censor) and '1's  (event)
        self.num_samples = len(y_data)
        self.y_data = y_data
        self.batch_size = batch_size
        self.Event_Inds = [i for i,k in enumerate(y_data[:,1]) if k==1.0]
        self.Non_Event_Inds = [i for i,k in enumerate(y_data[:,1]) if k==0.0]
        self.weights = torch.tensor([1.0 for k in range(y_data.shape[0])])
        self.default_order = False 
        
    def __iter__(self):
        if (self.default_order == False):
            
            #3. Okay, now we're going to randomly sample from the entire dataset
            Random_Indices = torch.multinomial(self.weights, self.num_samples, False)
            Num_Replacements_to_Prep = int(self.num_samples / self.batch_size) + 1
            
            #4. And we're also going to generate ceil(num_samples / batch_size)
            Replacement_Indices = torch.multinomial(torch.tensor([1.0 for k in self.Event_Inds]), Num_Replacements_to_Prep, True)
            Replacement_Indices = [self.Event_Inds[k] for k in Replacement_Indices]
            
            #5. Parse the 30k random indices we chose. If at any point we don't see an event for batch_size indices in a row, replace a non-event index with an event idnex
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
            
        # sometimes we want to return non-randomly (evaluation)
        else:
            yield from range(self.num_samples)

    def __len__(self):
        return self.num_samples
        
# %% Dataset Classes. PyCox needs everything frontloaded or done in the dataset (can't access the batch)
# Dataset_FuncList_XY returns x,y after applying [functions] to x
# Dataset_FuncList_X  returns x after applying [functions] to x

class Dataset_FuncList(torch.utils.data.Dataset):
    # Applies functions in func_list, in order, to x, 
    # Returns x, (y[0], y[1]). y[0] is time (float32), y[1] is event (0 or 1, int)
    # x,y must be Tensor for CollateFunc
    # ... Return_Toggle changes if just x, or x,y are returned
    
    def __init__(self, data, targets = None, func_list= None, discretize = False, Toggle = 'XY'):
        self.data = data
        
        if (targets is not None):
            if (discretize):
                self.targets = torch.tensor(targets.astype(np.int64)) # must be tensor and integer IF DISCRETIZING
            else:
                self.targets = torch.tensor(targets.astype(np.float64)) # must be tensor try not discretizing?
        else:
            self.targets = None
        self.func_list = func_list
        self.Return_Toggle = Toggle # 'X' - return only X. 'XY' - return both. PyCox formats

    def __getitem__(self, index): 
        if isinstance(index, slice): # ...LH fit requests 2 elements to check their sizes, as a slice... let's just do give it two samples?
            return [1,2,3] # seems to work for now? might break something!
            index = [0,1]
            
        x = self.data[index] # if you ever modify this value, torch.clone it first (else permanent)
        if (self.func_list is not None):
            x = torch.clone(x)
            for k in self.func_list:
                x = k(x)
                
        if (self.Return_Toggle =='XY'):
            y = self.targets[index]
            return x, (y[0], y[1].to(torch.float32)) #must be tensor, (int64tensor if discretizing, else float32/64), float32tensor. Outputs as tuple.
        
        if (self.Return_Toggle =='X'):
            return x, # must be a tensor, NEEDS A COMMA TO BE A TUPLE containing a tensor
    
    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack() # demands torch tensors


# % **********************************
# %%  ---------------- Start the model
class Generic_Model_PyCox(Generic_Model):

    def __init__(self, args, Data):
        # init should be overwritten at specific model level
        pass
    
# %% augment process_args, models should include this in init
    def Process_Args_PyCox(self, args):
        self.Process_Args(args)
        
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



# %% Gather what you need to normalize / discretize later
    def Prep_Data_Normalization_Discretization(self, args, Data):
        # Save data to self, 
        # if x_train, prep normalization and discretization 
        # if no x_train in Data, will load model and prep norm/discretization there
        # then we are loading a model and that loads normalization/discretization parameters)
        self.max_duration = None
        self.Data = Data
        if 'x_train' in Data.keys():
            self.Data['x_train'] = Structure_Data_NCHW(self.Data['x_train'])
            self.Prep_Normalization(args, self.Data)
            self.max_duration = max(Data['y_train'][:,0])
            self.labtrans = LogisticHazard.label_transform(self.num_durations)
            self.labtrans.fit_transform(np.array([0,self.max_duration]), np.array([0,1]))


# %% Overwrite Process_Data_To_Dataloaders
# This runs AFTER we have normalization and time discretization prepped from either init or load
# The goal is to get everything ready for PyCox model training
# 1) Prep which functions get called per image
# 2) Discretize or normalize Data
# 3) Prep Datasets and DataLoaders

    def Process_Data_To_Dataloaders(self):
        print('front-loading normalization')  
        
        func_list = []
        func_list.append(self.Adjust_Image) # adjust each input individually after loading
        
        if 'x_train' in self.Data.keys():
            self.Data['x_train'] = Structure_Data_NCHW(self.Data['x_train'])
            self.Data['x_train'] = Normalize(torch.Tensor(self.Data['x_train']), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            if (self.Discretize):
                self.Data['y_train'][:,0], self.Data['y_train'][:,1] = self.labtrans.transform(self.Data['y_train'][:,0], self.Data['y_train'][:,1])
            self.train_dataset = Dataset_FuncList(self.Data['x_train'] , targets = self.Data['y_train'], func_list = func_list, discretize=self.Discretize)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.GPU_minibatch_limit, collate_fn=collate_fn, sampler=Custom_Sampler(self.Data['y_train'], self.GPU_minibatch_limit)) 

        if 'x_valid' in self.Data.keys():
            self.Data['x_valid'] = Structure_Data_NCHW(self.Data['x_valid'])
            self.Data['x_valid'] = Normalize(torch.Tensor(self.Data['x_valid']), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            if (self.Discretize):
                self.Data['y_valid'][:,0], self.Data['y_valid'][:,1] = self.labtrans.transform(self.Data['y_valid'][:,0], self.Data['y_valid'][:,1])
            self.val_dataset  = Dataset_FuncList(self.Data['x_valid']  , targets = self.Data['y_valid'], func_list = func_list, discretize=self.Discretize )
            self.val_dataloader = torch.utils.data.DataLoader (self.val_dataset,  batch_size = self.GPU_minibatch_limit, collate_fn=collate_fn, sampler=Custom_Sampler(self.Data['y_valid'], self.GPU_minibatch_limit)) 

        if 'x_test' in self.Data.keys():
            self.Data['x_test'] = Structure_Data_NCHW(self.Data['x_test'])
            self.Data['x_test'] = Normalize(torch.Tensor(self.Data['x_test']), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            if (self.Discretize):
                self.Data['y_test'][:,0],  self.Data['y_test'][:,1]  = self.labtrans.transform(self.Data['y_test'][:,0],  self.Data['y_test'][:,1])
            self.test_dataset = Dataset_FuncList(self.Data['x_test'], targets = self.Data['y_test'], func_list = func_list, discretize=self.Discretize, Toggle='X') #Only returns X, not Y
            self.test_dataloader = torch.utils.data.DataLoader (self.test_dataset,  batch_size = self.GPU_minibatch_limit, collate_fn=collate_fn, shuffle = False) #DO NOT SHUFFLE
        

# %% Prep pycox model (here so as not to duplicate in train, load, and run)
    def Get_PyCox_Model(self):
        
        self.Prep_Optimizer_And_Scheduler()
        
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


# %% Overwrite Train
    def Train(self):
        # store a copy of the best model available
        Best_Model = copy.deepcopy(self.model)
        
        # Train
        train_loss = 0 
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
               
                # consider stopping
                val_perfs = np.array([k[2] for k in self.Perf])
                if (self.early_stop > 0):
                    if (len(val_perfs) - (np.argmin(val_perfs) + 1 ) ) >= self.early_stop:
                        # ^ add one: len_val_perfs is num trained epochs (starts at 1), but argmin starts at 0.
                        break
        self.model = copy.deepcopy(Best_Model)
        self.pycox_mdl.net = self.model
        return train_loss

# %% Overwrite Load
    def Load(self, best_or_last):
        
        print('load Should be overwritten in model file!')
        quit()
        
        # Import_Dict = self.Load_Checkpoint(best_or_last)
        # self.Load_Random_State(Import_Dict)
        # self.Load_Training_Params(Import_Dict)
        # self.Load_Training_Progress(Import_Dict)
        # self.Load_Normalization(Import_Dict)

        # # initialize model, update model shape, send to GPU, update weights, load optimizer and scheduler
        # Actual_output_class_count = Import_Dict['model_state_dict']['fc.weight'].shape[0]
        # self.model.fc = nn.Linear(in_features=512, out_features=Actual_output_class_count, bias=True)
        # self.model.to(self.device)
        # self.model.load_state_dict(Import_Dict['model_state_dict'])
        # if ('optimizer_state_dict' in Import_Dict.keys()):
        #     self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        # if ('scheduler_state_dict' in Import_Dict.keys()):
        #     self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
            
        # # Now set up time discretization
        # if ('max_duration' in Import_Dict.keys()):
        #     num_durations = Actual_output_class_count
        #     self.max_duration = Import_Dict['max_duration']
        #     self.labtrans = LogisticHazard.label_transform(num_durations)
        #     self.labtrans.fit_transform(np.array([0,self.max_duration]), np.array([0,1]))
        # else:
        #     print('cant discretize data')
            
        # # Now frontload normalization (CPU) and discretize times
        # if (self.max_duration is not None):
        #     self.Process_Data_To_Dataloaders()

    def Discretize_On_Load(self, Import_Dict):
        if ('max_duration' in Import_Dict.keys()):
            if (self.Num_Classes > 1): # If we're loading an LH/MTLR/DeepHit model, overwrite duration count
                self.num_durations = self.Num_Classes
            self.max_duration = Import_Dict['max_duration']
            self.labtrans = LogisticHazard.label_transform(self.num_durations)
            self.labtrans.fit_transform(np.array([0,self.max_duration]), np.array([0,1])) #dunno why, but it wanted more examples
        else:
            print('cant discretize data')
            
        # Now frontload normalization (CPU) and discretize times (ONLY if you haven't already - if Training, you have)!
        if (self.max_duration is not None):
            if ( ('Train' not in self.args.keys()) or (self.args['Train'] == 'False') ):
                self.Process_Data_To_Dataloaders()

# %% Overwrite Run, include output discretization from continuous models
    def Run_NN (self, my_dataloader):

        if (self.args['pycox_mdl'] == 'CoxPH'):
            self.pycox_mdl.compute_baseline_hazards(input=self.Adjust_Many_Images(self.Data['x_train'][:,0,:,:]),target=[self.Data['y_train'][:,0],self.Data['y_train'][:,1]],batch_size = self.GPU_minibatch_limit)
           
        surv    = self.pycox_mdl.predict_surv(my_dataloader)
        surv_df = self.pycox_mdl.predict_surv_df(my_dataloader) # contains surv, also stores 'index' which is the time in years rather than discretized points

        # for CoxPH you have to discretize time points
        if (self.args['pycox_mdl'] == 'CoxPH'):
            # survival (x) is probability that event occurs AFTER the current time point (surv[:,0] != 1)
            # so ... we have 738 indices from train we want to map to 100 cuts
            # ... but they only fall into 90 cut 'slots'
            # So: 
            # parse columns of output: [large] x 100
            # parse columns of input: [large] x 738. 
            # input columns align with output columns via 't2': ex: array([ 1,  1,  2,  3,  6, 10, 10]). 90 unique values.
            # where an input maps to an output, fill that in, if there's a gap, fill in prev known value
            Unique_Time_Points = np.unique(self.Data['y_train'][:,0])
            t2, k = self.labtrans.transform(Unique_Time_Points, np.ones(Unique_Time_Points.shape))
            surv_out = np.ones( (my_dataloader.dataset.targets.shape[0],len(self.labtrans.cuts)), dtype=float)
            temp_col = surv[:,0]
            for k in range(surv_out.shape[1]):  # parse output columns k
                for i,m in enumerate(t2):       # parse input columns i
                    if (m > k):                 # if i is associated with a later k, break loop
                        break
                    if (m==k):                  # if i is associated with k, remember input col
                        temp_col = surv[:,i]    # [this repeats until we get the last input col associated with k]
                surv_out[:,k] = temp_col       # after parsing all input columns m, we have the highest m <= k

            surv = surv_out
            t, d = self.labtrans.transform(my_dataloader.dataset.targets[:,0].numpy(),my_dataloader.dataset.targets[:,1].numpy())
            
            # surv transposed in df 5/25/24
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
    
# %% Overwrite Save
    
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
    
    Out_Dict['best_performance_measure'] = best_performance_measure
    if (optimizer is not None):
        Out_Dict['optimizer_state_dict'] = optimizer.state_dict()
        
    if (scheduler is not None):
        Out_Dict['scheduler_state_dict'] = scheduler.state_dict()
        
    # Normalization Parameters
    if (NT is not None):
        Out_Dict['NT'] = NT
        
    if (NM is not None):
        Out_Dict['NM'] = NM
        
    if (NS is not None):
        Out_Dict['NS'] = NS
        
    # Time discretization - if you have the number of cuts (which you get from model size) and the max duration, you can recreate the discretization
    if (max_duration is not None):
        Out_Dict['max_duration'] = max_duration
    
    torch.save(Out_Dict, path)
    
