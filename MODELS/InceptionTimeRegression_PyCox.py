# InceptionTimeRegression_PyCox is a Generic_Model_PyCox, which in turn is a Generic_Model
# args['K_M'] defaults to 1. Scales kernel widths from default of [10,20,40] 

# adapted from
# https://github.com/TheMrGhostman/InceptionTime-Pytorch/commit/2a1b81d82b624c9033e73a2e7ee9bba2414217ef
# Heavily modified by PVL to line up with the rest of the survival modeling flow
# Where possible, the original code was kept

import numpy as np 
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler

from MODELS.InceptionTime_Support import Inception, InceptionBlock


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
        
def InceptionTime(in_channels, Kernel_Mult = 1):
    # *** MODEL
    
    Kernel_Width_Base = [10,20,40]
    InceptionTime = nn.Sequential(
                        # Reshape(out_shape=(1,4096)), # I'm not sure this is necessary? PVL 12/14/23
                        InceptionBlock(
                            in_channels=in_channels, 
                            n_filters=32, 
                            kernel_sizes=[(k*Kernel_Mult)+1 for k in Kernel_Width_Base],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        InceptionBlock(
                            in_channels=32*4, 
                            n_filters=32, 
                            kernel_sizes=[(k*Kernel_Mult)+1 for k in Kernel_Width_Base],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        nn.AdaptiveAvgPool1d(output_size=1),
                        nn.Flatten(),
                        # Flatten(out_features=32*4*1),
                        nn.Linear(in_features=4*32*1, out_features=1)
            )
    
    return InceptionTime

class InceptionTimeRegression_PyCox(Generic_Model_PyCox):

    def __init__(self, args, Data):
        # Generic_Model_PyCox init: 1) Process Args 
        self.Process_Args_PyCox(args) # process common arguments, determine number of output classes
        
        # Extra processing on args
        if ('K_M' in args.keys()):
            self.K_M = int(args['K_M'])
        else:
            self.K_M = 1
            print('defaulting to kernel mult of 1. InceptionTime kernel sizes are 10,20,40')

        # Generic_Model_PyCox init: 2) Prep Normalization 
        self.Prep_Data_Normalization_Discretization(args, Data) # sets num_durations, frontloads normalization and discretization, self.Data = Data
        
        # Generic_Model_PyCox init: 3) Prep DataLoaders
        if (self.max_duration is not None):
            self.Process_Data_To_Dataloaders()
            
        # Generic_Model_PyCox init: 4) Prep a model, send to device
        if 'x_train' in Data.keys():
            in_channels = Data['x_train'].shape[-1]
            self.model = InceptionTime(in_channels=in_channels, Kernel_Mult = self.K_M)
            if 'y_train' in Data.keys():
                self.model[-1] = nn.Linear(in_features=4*32*1, out_features=self.Num_Classes, bias=True) # self.Num_Classes is set in Generic_Model_PyCox
            self.model.to(self.device)
            self.pycox_mdl = self.Get_PyCox_Model() # init the optimizer and scheduler

        

# %%
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)
        
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        # self.Load_Normalization(Import_Dict) # we frontload normalization based on Train data, so this no longer matters

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
        single_image = torch.transpose(single_image, 1,2)
        return single_image[0] # Just chan x leng, so 12 x 4k
    
    def Adjust_Many_Images(self, many_images):
        many_images = torch.transpose(many_images, 1,2)
        return many_images # Just chan x leng, so 12 x 4k