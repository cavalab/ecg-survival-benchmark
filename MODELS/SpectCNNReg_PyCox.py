# InceptionTimeRegression_PyCox is a Generic_Model_PyCox, which in turn is a Generic_Model
# args['K_M'] defaults to 1. Scales kernel widths from default of [10,20,40] 

# adapted from
# https://github.com/TheMrGhostman/InceptionTime-Pytorch/commit/2a1b81d82b624c9033e73a2e7ee9bba2414217ef
# Heavily modified by PVL to line up with the rest of the survival modeling flow
# Where possible, the original code was kept

import torch
import os
from tqdm import tqdm

import torchaudio

import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from MODELS.GenericModel import GenericModel

# import Support_Functions
from MODELS.Support_Functions import Custom_Dataset
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss


# %%
import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from MODELS.Generic_Model_PyCox import Generic_Model_PyCox
from MODELS.Support_Functions import Structure_Data_NCHW
import torchtuples as tt
from pycox.models import LogisticHazard
        
class Spect_CNN(nn.Module):
    # just wrapping the LSTM here to return the correct h/c outputs, not the outputs per ECG time point
    # https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7
    
    def __init__ (self, output_size):
        super(Spect_CNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(12, 32, (4,4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Conv2d(64, 64, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features = 246016, out_features = output_size)
            )
        
        self.Spect = torchaudio.transforms.MelSpectrogram(sample_rate = 400, n_mels = 512, n_fft=1024, hop_length=8).to('cuda')
        

    def forward(self, input_ecg):
        # hmm. looks like PyCox adds a squeeze in dataloader outputs.
        # ... but only when training. 
        
        a = self.Spect( torch.transpose( input_ecg,2,1)) # NCHW 32 x 12 x 512 x 513
        ret = self.model(a[:,:,:,:512]) # cut last freq to line up size
        return ret # output is N x output_shape


class SpectCNNReg_PyCox(Generic_Model_PyCox):
    def __init__(self, args, Data):
        
        
        # Generic_Model_PyCox init: 1) Process Args 
        self.Process_Args_PyCox(args) # process common arguments, determine number of output classes

        # Generic_Model_PyCox init: 2) Prep Normalization 
        self.Prep_Data_Normalization_Discretization(args, Data) # sets num_durations, frontloads normalization and discretization, self.Data = Data
        
        # Generic_Model_PyCox init: 3) Prep DataLoaders
        if (self.max_duration is not None):
            self.Process_Data_To_Dataloaders()
            
        # Generic_Model_PyCox init: 4) Prep a model, send to device
        
        self.model = Spect_CNN(self.Num_Classes)
        
        self.model.to(self.device)
        self.pycox_mdl = self.Get_PyCox_Model() # init the optimizer and scheduler

        

# %%
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)
        
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        self.Load_Normalization(Import_Dict) # we frontload normalization based on Train data, so this no longer matters

        # initialize model, update model shape, send to GPU, update weights, load optimizer and scheduler
        # Actual_output_class_count = Import_Dict['model_state_dict']['4.weight'].shape[0]
        # if ('0.inception_1.bottleneck.weight' in Import_Dict['model_state_dict'].keys()):
        #     input_chan_count = Import_Dict['model_state_dict']['0.inception_1.bottleneck.weight'].shape[1]
        # else:
        #     input_chan_count = 1
            
        # self.model = InceptionTime(in_channels=input_chan_count, Kernel_Mult = self.K_M)
        # self.model[-1]  = nn.Linear(in_features = 4*32*1, out_features=Actual_output_class_count, bias=True)
        # self.model.to(self.device)
        # self.pycox_mdl = self.Get_PyCox_Model() # init the optimizer and scheduler
        self.model.load_state_dict(Import_Dict['model_state_dict'])
        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
            print('loaded optimizer')
        else:
            print('NO optimizer loaded')
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
            print('loaded scheduler')
        else:
            print("NO scheduler loaded")
            
            
        # Now set up time discretization
        # self.Discretize_On_Load(Import_Dict)
        self.Load_Random_State(Import_Dict)

                
    # %%Overwrite how we adjust _each_ input and _multiple_ inputs. InceptionTime wants a different shape for data (not NCHW but NHW)
    def Adjust_Image(self, single_image):
        # single_image = torch.transpose(single_image, 1,2)
        return single_image[0] # Just chan x leng, so 12 x 4k
        # return single_image
    
    def Adjust_Many_Images(self, many_images):
        # many_images = torch.transpose(many_images, 1,2)
        return many_images # Just chan x leng, so 12 x 4k