"""
This is the framework for a generic model.

# args['epoch_start']           int. defaults to 0
# args['epoch_end']             int. defaults to -1 (no training)
# args['validate_every']        int. defaults to 1
# args['batch_size']            int. Size of batch. Defaults to 512
# args['GPU_minibatch_limit']   int. Size of sub-batch sent to GPU. Defaults to 64. (gradients accumulate to batch_size)
# args['Save_Out_Checkpoint']   'True' or not. defaults to 'True'. iff 'True', saves out model checkpoint after every epoch
# args['Save_Out_Best']         'True' or not. defaults to 'True'. iff 'True', saves out best checkpoint after validation
# args['Model_Folder_Path']     '<path>'. No default. Necessary. Passed from Runner scripts.
# args['Loss_Type']             'SSE' or 'CrossEntropyLoss' or 'wSSE' or 'SAE' or 'wSAE'. Defaults to 'SSE'. GenericModelPyCox overwrites this.
# args['early_stop]             int. defaults to -1. 
# args['optimizer']             str. only option, and default, here, is 'adam'
# args['Adam_wd']               float. defaults to 0.01.
# args['Scheduler']             'True' to set 1e-8 minimum learning rate (scheduler scales 1e-7 by 0.1x). Defaults to 1e-3 (scheduler scales default 1e-2 by 0.1x to get 1e-3 rate). mults rate by 0.1x if no val improvement in 10E. Stops early if at minimum


Has generic functions, oriented around regressions/classifiers, to:
    1) Processes passed args
    2) Prepare normalization parameters (normalization is not front-loaded in this script, it is done per-batch during train/run)
    3) Prepare loss functions
    4) Initialize optimizers and schedulers
    5) Prepare dataloaders 
    6) Load: checkpoint, random state, best model, training parameters, training progress
    6) Adjust inputs: one at a time and in a batch- these are placeholders and would be overwritten if a model needs data in e.g. a different shape than the N-C-H-W we're standardizing to
    7) Train a model
    8) Run a model on a dataloader
    9) Storing training/validation performance
    
    
A specific_model would look like this:
    
class Specific_Model(Generic_Model):
    def __init__(self, args, Data):
        self.Process_Args(args)
        // process additional model-specific arguments
        //set up model
        self.Process_Data_To_Dataloaders(Data)
        self.Prep_Normalization(args, Data)
        self.Prep_LossFunction(args, Data)
    
    def load(self, best_or_last):
        self.Load_Checkpoint(self, best_or_last)
        // initialize model, update model shape, send to GPU, update weights, load optimizer and scheduler
        self.Load_Random_State(self, Import_Dict)
        self.Load_Training_Params(self, Import_Dict)
        self.Load_Training_Progress(self, Import_Dict)
        self.Load_Normalization(self, Import_Dict)

Unresolved bug: 
    # train 2 epochs != train 1 epoch, save, load, train 1 epoch. 
    # the random state saved and loaded is correct
    # ... it looks like dataloaders that shuffle inputs have their own states that need to be saved
    # ... that's not implemented yet

"""

import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

# import Support_Functions
from MODELS.Support_Functions import Custom_Dataset
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss

# LSTM support functions
# from MODELS.Support_Functions import CustomDatasetLSTM
# from MODELS.Support_Functions import CustomSamplerLSTM
# from MODELS.Support_Functions import Custom_Collate_LSTM
        
class GenericModel:
    
    def __init__(self): # doesn't automatically init; the specific_model should call/overwrite/extend the generic functions here
        pass
    
    # %% Init components
    def Process_Args(self, args):
        self.args = args
        
        # CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Set to Run on ' + str(self.device))
        
        # Optimizer
        if ('optimizer' in args.keys()):
            self.optimizer_name = str(args['optimizer'])
        else:
            self.optimizer_name = 'Adam'
            print('GenericModel: By default, using adam optimizer')

        if (self.optimizer_name == 'Adam'):
            if ('Adam_wd' in args.keys()):
                self.Adam_wd = float(args['Adam_wd'])
            else:
                self.Adam_wd = 0.01 # default weight decay on Adam
        
        # Epochs and Validation
        if ('epoch_start' in args.keys()):
            self.epoch_start = int(args['epoch_start'])
        else:
            self.epoch_start = 0
            print('GenericModel: By default, starting at epoch 0')
            
        if ('epoch_end' in args.keys()):
            self.epoch_end = int(args['epoch_end'])
        else:
            self.epoch_end = -1
            print('GenericModel: By default, ending after 0 training epochs')
            
        if ('validate_every' in args.keys()):
            self.validate_every = int(args['validate_every'])
        else:
            self.validate_every = 1
            print('GenericModel: By default, validating every epoch')
            
        if ('early_stop' in args.keys()):
            self.early_stop = int(args['early_stop'])
        else:
            self.early_stop = -1
            print('GenericModel: By default, no early stopping (args[early_stop]=-1)')
            
        # Batch Size
        if ('batch_size' in args.keys()):
            self.batch_size = int(args['batch_size'])
        else:
            self.batch_size = 512
            print('GenericModel: By default, using batch size of 512')
            
        if ('GPU_minibatch_limit' in args.keys()):
            self.GPU_minibatch_limit = int(args['GPU_minibatch_limit'])
        else:
            self.GPU_minibatch_limit = 128
            print('GenericModel: By default, using GPU split-batch limit of 128')

        if (self.batch_size % self.GPU_minibatch_limit !=0):
            print('GenericModel: GPU_minibatch_limit size not factor of batch size! adjust!')
            
        # Run Params
        if ('Save_Out_Checkpoint' in args.keys()):
            self.Save_Out_Checkpoint = (args['Save_Out_Checkpoint'] == 'True')
        else:
            self.Save_Out_Checkpoint = True
            print('GenericModel: By default, checkpointing model')
            
        if ('Save_Out_Best' in args.keys()):
            self.Save_Out_Best = (args['Save_Out_Best'] == 'True')
        else:
            self.Save_Out_Best = True
            print('GenericModel: By default, Saving out best validation model')
            
        # Scheduler
        if ('Scheduler' not in args.keys()):
            args['Scheduler'] = 'None'
            print('GenericModel: By default, No Scheduler')
            self.min_lr = 1e-2 # scaled by 1e-1 later
            
        else:
            if (args['Scheduler'] == 'True'):
                self.min_lr = 1e-7 # scaled by 1e-1 later
            else:
                self.min_lr=1e-2 # scaled by 1e-1 later
                
                
        # Is this an lstm?
        self.LSTM = False
        if ('LSTM' in args.keys()):
            if args['LSTM'] == 'True':
                self.LSTM = True
            

        # Model Folder Path
        self.model_folder_path = args['Model_Folder_Path']
        
        
        # Prep training parameters (that can be overwritten by load() )
        self.Val_Best_Loss = 9999999
        self.Perf = []
        
    # %% Process Data - normalize, convert to tensors
    def Prep_Dataloaders_and_Normalize_Data(self):
        a = time.time()
        if 'x_train' in self.Data.keys():
            self.Data['x_train']  = Structure_Data_NCHW(self.Data['x_train'])
            self.Data['x_train']  = Normalize(torch.Tensor(self.Data['x_train']), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            self.Data['y_train']  = np.float64(self.Data['y_train']) #8/6/24
            self.train_dataset = Custom_Dataset( self.Data['x_train'] , self.Data['y_train'][:,-1]) # modified event, e*, lives in -1
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.GPU_minibatch_limit, shuffle = True) # weighted sampler is mutually exclussive with shuffle = True
            
        if 'x_valid' in self.Data.keys():
            self.Data['x_valid']  = Structure_Data_NCHW(self.Data['x_valid'])
            self.Data['x_valid']  = Normalize(torch.Tensor(self.Data['x_valid']), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            self.Data['y_valid']  = np.float64(self.Data['y_valid']) #8/6/24
            self.val_dataset = Custom_Dataset(self.Data['x_valid']  , self.Data['y_valid'][:,-1]) # modified event, e*, lives in -1
            self.val_dataloader = torch.utils.data.DataLoader (self.val_dataset,  batch_size = self.GPU_minibatch_limit, shuffle = False) #DO NOT SHUFFLE

        if 'x_test' in self.Data.keys():
            self.Data['x_test'] = Structure_Data_NCHW(self.Data['x_test'])
            self.Data['x_test']  = Normalize(torch.Tensor(self.Data['x_test']), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            self.Data['y_test']  = np.float64(self.Data['y_test']) #8/6/24
            self.test_dataset  = Custom_Dataset(self.Data['x_test']  , self.Data['y_test'][:,-1]) # modified event, e*, lives in -1
            self.test_dataloader = torch.utils.data.DataLoader (self.test_dataset,  batch_size = self.GPU_minibatch_limit, shuffle = False) #DO NOT SHUFFLE
        print('GenericModel: Dataloader prep T = ', time.time() - a)


    def prep_normalization_and_reshape_data(self, args, Data):
        a = time.time()
        self.Data = Data
        # need to reshape x_train now that we're frontloading normalization
        self.Data['x_train'] = Structure_Data_NCHW(self.Data['x_train'])
        
        if 'x_train' in Data.keys():            
            self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev = Get_Norm_Func_Params(args, Data['x_train'])
        else:
            print('Cant prep normalization')
        print('GenericModel: Train Data reshape and normalization prep T = ', time.time() - a)

    def Prep_LossFunction(self, args, Data):
        if 'y_train' in Data.keys():     
            self.Loss_Params = Get_Loss_Params(args, Train_Y = Data['y_train'])
        elif ('Loss_Type' in args.keys()):
            self.Loss_Params = Get_Loss_Params(args) 
        else:
            args['Loss_Type'] = 'SSE'
            print ('By default, using SSE loss')
            self.Loss_Params = Get_Loss_Params(args) 

    def Prep_Optimizer_And_Scheduler(self):
        
        if (self.optimizer_name == 'cocob'):
            from parameterfree import COCOB
            self.optimizer = COCOB(self.model.parameters(),weight_decay = self.cocob_wd) 
        else:
            print('GenericModel: By default, using optimizer: AdamW 1e-3')
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3, weight_decay = self.Adam_wd)
            
            # add the ribeiro scheduler (ineffective unless args['Scheduler'] == 'True')
            self.patience = 10
            self.lr_factor = 0.1
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience,
                                                             min_lr=self.lr_factor * self.min_lr,
                                                             factor=self.lr_factor)
        
        if ((self.args['Scheduler'] == 'True') and (self.optimizer_name != 'Adam')):
            print('scheduler is not compatible with parameterfree optimizers')
            quit()
            
            
    # %% 'Load' components
            
    def Load_Checkpoint(self, best_or_last):
        if (best_or_last == 'Last'):
            some_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
            
        elif (best_or_last == 'Best'):
            some_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
            
        else:
            print('model load failed. Specify --Load Best or --Load Last')
            quit()
        
        Import_Dict = torch.load(some_path) # load a checkpoint
        print('Loaded model from: '+ some_path)
        
        return Import_Dict
    
    def Load_Random_State(self, Import_Dict):
        if ('Numpy_Random_State' in Import_Dict.keys()):
            if ('Torch_Random_State' in Import_Dict.keys()):
                np.random.set_state(Import_Dict['Numpy_Random_State'])
                torch.random.set_rng_state(Import_Dict['Torch_Random_State'])
                if ('CUDA_Random_State' in Import_Dict.keys()):
                    torch.cuda.random.set_rng_state(Import_Dict['CUDA_Random_State'])
                else:
                    print('CUDA RANDOM STATE NOT LOADED. Further training not deterministic')
                torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
            else:
                print('Could not load random state')
        else:
            print('Could not load random state')
            
    def Load_Training_Params(self, Import_Dict):
        if ('epoch' in Import_Dict.keys()):
            self.epoch_start = Import_Dict['epoch']+1
        else:
            print('Could not load last epoch count')
        
        if ('best_performance_measure' in Import_Dict.keys()):
            self.Val_Best_Loss = Import_Dict['best_performance_measure']
        else:
            print('Could not load validation measure')
        
    def Load_Training_Progress(self, Import_Dict):
        csv_file_path = os.path.join(self.model_folder_path, 'Training_Progress.csv')
        if (os.path.isfile(csv_file_path)):
            self.Perf = np.genfromtxt(csv_file_path, skip_header=1, delimiter=",") # top row is header, so skip
            self.Perf = self.Perf.tolist()
            if (type(self.Perf[0]) != list): # If just 1 epoch, get values. want list of values.
                self.Perf = [self.Perf]
        else:
            print('Could not load Training_Progress.csv')
            
    def Load_Normalization(self, Import_Dict):
        if ('NT' in Import_Dict.keys()):
            self.Normalize_Type  = Import_Dict['NT']
            self.Normalize_Mean  = Import_Dict['NM']
            self.Normalize_StDev = Import_Dict['NS']
        else:
            print('Could not load normalization data. Assuming no normalization step.')
            self.Normalize_Type  = 'No_Norm'
            self.Normalize_Mean  = 0
            self.Normalize_StDev = 0
    
    
        
# %% Adjust each ECG as it comes in. Importantly, this is on CPU!
    def Adjust_Image(self, single_image):
        # Keep in mind that pycox
        # Normalization and Tensor conversion now frontloaded (CPU) for speed
        return single_image
    
# %% Adjust many ECGs, on CPU, for model calibration
    def Adjust_Many_Images(self, image_batch):
        # Keep in mind that pycox
        # Normalization and Tensor conversion now frontloaded (CPU) for speed
        return image_batch    
     

    # %% tuned for classifier
    def Train(self):
        Best_Model = copy.deepcopy(self.model)
        
        train_loss = -1 # in case no training occurs
        accumulation_iterations = int(self.batch_size / self.GPU_minibatch_limit)
        
        # Try to load a checkpointed model?
        if self.epoch_end > self.epoch_start:
            print('GenericModel.Train(): Training Requested. Loading best then last checkpoints.')
            last_checkpoint_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
            best_checkpoint_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
            if (os.path.isfile(last_checkpoint_path)):
                if (os.path.isfile(best_checkpoint_path)):
                    self.Load('Best')
                    Best_Model = copy.deepcopy(self.model)
                    print('GenericModel.Train(): Best Checkpoint loaded and Best model copied.')
                    self.Load('Last')
                    print('GenericModel.Train(): Checkpointed model loaded. Will resume training.')
                    
                    val_perfs = np.array([k[2] for k in self.Perf])
                    if (self.early_stop > 0):
                        if (len(val_perfs) - (np.argmin(val_perfs) + 1 ) ) >= self.early_stop:
                            # ^ add one: len_val_perfs is num trained epochs (starts at 1), but argmin starts at 0.
                            print('GenericModel.Train(): Model at early stop. Setting epoch_start to epoch_end to cancel training')
                            self.epoch_start = self.epoch_end
                            
                    if (self.epoch_start == self.epoch_end):
                        print('GenericModel.Train(): Loaded checkpointed model already trained')
                    if (self.epoch_start > self.epoch_end):
                        print('GenericModel.Train(): Requested train epochs > current epochs trained. evaluating.')
                        self.epoch_start = self.epoch_end
                        # breakpoint()
                else:
                    print('GenericModel.Train(): FAILED to load best model! Eval may be compromised')
            else:
                print('GenericModel.Train(): Last checkpoint unavailable.')
        
        
        for epoch in range(self.epoch_start, self.epoch_end):
            self.model.train()
            epoch_start_time = time.time()
            
            train_loss = 0
            for i, (imgs , labels) in enumerate(self.train_dataloader):

                imgs = imgs.to(self.device)
                imgs = imgs.to(torch.float32) # convert to float32 AFTER putting on GPU
                # imgs = Normalize(imgs, self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
                imgs = self.Adjust_Many_Images(imgs)
                
                labels = labels.to(self.device)
                labels = labels.type(torch.float32) # again, convert type AFTER putting on GPU
                

                model_out = self.model(imgs)
                
                
                loss = Get_Loss(model_out, labels, self.Loss_Params) 
                train_loss += loss.item()
                loss.backward()
                
                # minibatch implem - from https://kozodoi.me/blog/20210219/gradient-accumulation
                if  ( ((i + 1) % accumulation_iterations ==0) or (i+1) == len(self.train_dataloader)):
                    self.optimizer.step() 
                    self.model.zero_grad()  

            epoch_end_time = time.time()

            # ----
            # Run Validation and Checkpoint
            if ( (epoch+1) % self.validate_every ==0):
                
                outputs, val_loss, correct_outputs = self.Run_NN(self.val_dataloader)
                
                # update scheduler # no effect unless args['Scheduler'] == 'True'
                if (hasattr(self,'scheduler')):
                    self.scheduler.step(val_loss)
                    tmp_LR = self.optimizer.state_dict()['param_groups'][0]['lr']
                else:
                    tmp_LR = 0
                    
                # If this is the new best model, save it as the best model
                if val_loss < self.Val_Best_Loss: 
                    nn_file_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
                    if (self.Save_Out_Best):
                        Save_NN(epoch, self.model, nn_file_path, optimizer=None, scheduler=None, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev)
                        Best_Model = copy.deepcopy(self.model) # store a local copy of the model
                    Save_Train_Args(os.path.join(self.model_folder_path,'Train_Args.txt'), self.args)
                    self.Val_Best_Loss = val_loss
                    
                # And checkpoint model in any case
                nn_file_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
                if (self.Save_Out_Checkpoint):
                    if (hasattr(self,'scheduler')):
                        Save_NN(epoch, self.model, nn_file_path, optimizer = self.optimizer, scheduler=self.scheduler, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev)
                    else:
                        Save_NN(epoch, self.model, nn_file_path, optimizer = self.optimizer, scheduler=None, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev)
                Save_Train_Args(os.path.join(self.model_folder_path,'Train_Args.txt'), self.args)

                # calculate new performance
                new_perf = [epoch, train_loss, val_loss, tmp_LR, epoch_end_time - epoch_start_time]
                print(new_perf)
                self.Perf.append(new_perf)
                
                # update performance                
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
                     
        # now that we're done training, load the best model back for evaluation
        self.model = copy.deepcopy(Best_Model)
        return train_loss
    
    # %% Run
    def Run_NN (self, my_dataloader):
        # Runs the net in eval mode, predicted output, loss, correct output
        self.model.eval()
        
        tot_loss = 0
        
        outputs = []
        correct_outputs = [] 
        
        for i, (imgs , labels) in enumerate(my_dataloader):
            imgs = imgs.to(self.device)
            imgs = imgs.to(torch.float32) # convert to float32 AFTER putting on GPU
            # imgs = Normalize(imgs, self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            imgs = self.Adjust_Many_Images(imgs)
            
            labels = labels.to(self.device)
            labels = labels.type(torch.float32) # again, convert type AFTER putting on GPU
        
            with torch.no_grad():
                model_out = self.model(imgs)
            
            loss = Get_Loss(model_out, labels, self.Loss_Params) 
        
            tot_loss += loss.item()
            outputs = outputs + model_out.to("cpu").detach().tolist()
            correct_outputs = correct_outputs + labels.to("cpu").detach().tolist()
        
        return outputs, tot_loss, correct_outputs
    
    # %% save out performance curves
    def Save_Perf_Curves (self,Perf, Path):
        temp = np.array(Perf)
        if (len(temp.shape)==1):
            temp = np.expand_dims(temp, axis=0)
            
        x = temp[:,0]
        y_train = temp[:,1]
        y_valid = temp[:,2]
        fig = plt.plot(x,y_train, '-o',color='r')
        plt.plot(x,y_valid, '-o',color='b')
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(Path)
    
    # %% Test
    def Test(self, Which_Dataloader = 'Test'):
        if (Which_Dataloader == 'Train'):            
            self.train_dataloader.batch_sampler.shuffle = False
            outputs, test_loss, correct_outputs = self.Run_NN(self.train_dataloader) # NOT shuffled
            self.train_dataloader.batch_sampler.shuffle = True
        elif (Which_Dataloader == 'Validation'):
            outputs, test_loss, correct_outputs = self.Run_NN(self.val_dataloader) # NOT shuffled
        else:
            outputs, test_loss, correct_outputs = self.Run_NN(self.test_dataloader) # NOT shuffled
        return outputs, test_loss, correct_outputs