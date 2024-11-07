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
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss

        
# %% The generic net (encode / decode)
class FusionModel(nn.Module):
    # We encode the ECG with some passed model
    # We process covariates with a linear model
    # Then we combine them with a fusion model
    # ... or we just decode the ECG with a linear layer
    
    def __init__(self, ECG_Model, out_classes, fusion_layers, fusion_dim, cov_layers, cov_dim):
        # inputs:
        # ECG_Model: a CNN or something that processes ECG to features (encoder)
        # out_classes: number of output classes from the fusionmodel
        # direct: if direct, only adds a linear layer between features and out_classes
        #     if indirect,adds 2 linear/relu chunks first 

        
        super(FusionModel, self).__init__()
        # ECG processing chunk
        self.ECG_Model = ECG_Model
        
        # Covariate processing chunk
        self.covariate_module_list = nn.ModuleList()
        for k in range(cov_layers):
            self.covariate_module_list.append(nn.LazyLinear(out_features=cov_dim))
            self.covariate_module_list.append(nn.ReLU())
        
        # Fusion chunk and final linear chunk
        self.fusion_module_list = nn.ModuleList()
        for k in range(fusion_layers):
            self.fusion_module_list.append(nn.LazyLinear(out_features=fusion_dim))
            self.fusion_module_list.append(nn.ReLU())
            
        self.fusion_module_list.append(nn.LazyLinear(out_features=out_classes))
        
        # extra variables in case we want to return the '-1' layer
        self.return_second_to_last_layer = False
        
    def forward(self, x, z):
        # x - ECG
        # z - covariates - float32
        
        # process ECG
        a = self.ECG_Model(x) 
        
        # process covariates and append. 
        # only happens if 1) covariates provided and 2) covariate modules initialized
        if (z.shape[1] > 0):
            if (len(self.covariate_module_list) > 0):
                b = z
                for covariate_layer in self.covariate_module_list:
                    b = covariate_layer(b)
                a = torch.concatenate( (a,b),dim=1)
        
        # apply fusion layers (includes final linear layer)
        for i,fusion_layer in enumerate(self.fusion_module_list):
            
            # if you want to return the second-to-last layer, do so here
            if (i == len(self.fusion_module_list) - 1 ): # when on last layer
                if(self.return_second_to_last_layer): 
                    return a
                
            # otherwise keep applying layers
            a = fusion_layer(a)
            
        return a

    
# %%
class GenericModel:
    def __init__(self): # doesn't automatically init; generic_model_x should call/overwrite/extend the generic functions here
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

        # Model Folder Path
        self.model_folder_path = args['Model_Folder_Path']
        
        # Prep training parameters (that can be overwritten by load() )
        self.Val_Best_Loss = 9999999
        self.Perf = []
        
    # %% Data pieces
    def restructure_data(self):
        a = time.time()
        for key in ['x_train', 'x_valid', 'x_test']:
            if key in self.Data.keys() :
                self.Data[key] = Structure_Data_NCHW(self.Data[key])
        print('GenericModel: restructure_data T = ', '{:.2f}'.format(time.time()-a))
                
    def prep_normalization_parameters(self):
        a = time.time()
        if 'x_train' in self.Data.keys():            
            self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev = Get_Norm_Func_Params(self.args, self.Data['x_train'])
        else:
            print('Cant prep normalization')
        print('GenericModel: prep_normalization_parameters T = ', '{:.2f}'.format(time.time()-a))
        
    def normalize_data(self):
        a = time.time()
        for key in ['x_train', 'x_valid', 'x_test']:
            if key in self.Data.keys() :
                self.Data[key] = Normalize(torch.Tensor(self.Data[key]), self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
        print('GenericModel: normalize_data T = ', '{:.2f}'.format(time.time()-a))
        
# %% Network pieces
    # %% Fusion pieces
    def prep_fusion(self, out_classes = 2):
        
        # Process Fusion args
        if ('fusion_layers' in self.args.keys()):
            fusion_layers = int(self.args['fusion_layers'])
        else:
            fusion_layers = 0 # go direct to output
            
        if ('cov_layers' in self.args.keys()):
            cov_layers = int(self.args['cov_layers'])
        else:
            cov_layers = 0
            
        if ('fusion_dim' in self.args.keys()):
            fusion_dim = int(self.args['fusion_dim'])
        else:
            fusion_dim = 128
            
        if ('cov_dim' in self.args.keys()):
            cov_dim = int(self.args['cov_dim'])
        else:
            cov_dim = 32
            
        self.model = FusionModel(self.model, out_classes, fusion_layers, fusion_dim, cov_layers, cov_dim)
        self.model.to(self.device)
        
    def prep_classif_loss(self):
        if 'y_train' in self.Data.keys():     
            self.Loss_Params = Get_Loss_Params(self.args, Train_Y = self.Data['y_train'])
        elif ('Loss_Type' in self.args.keys()):
            self.Loss_Params = Get_Loss_Params(self.args) 
        else:
            self.args['Loss_Type'] = 'SSE'
            print ('By default, using SSE loss')
            self.Loss_Params = Get_Loss_Params(self.args) 

    def prep_optimizer_and_scheduler(self):
        
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
        
    