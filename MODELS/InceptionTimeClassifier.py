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

from MODELS.Generic_Model import Generic_Model

# import Support_Functions
from MODELS.Support_Functions import Custom_Dataset
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss
        
def InceptionTime(in_channels):
    # *** MODEL
    InceptionTime = nn.Sequential(
                        # Reshape(out_shape=(1,4096)), # I'm not sure this is necessary? PVL 12/14/23
                        InceptionBlock(
                            in_channels=in_channels, 
                            n_filters=32, 
                            kernel_sizes=[11, 21, 41],
                            bottleneck_channels=32,
                            use_residual=True,
                            activation=nn.ReLU()
                        ),
                        InceptionBlock(
                            in_channels=32*4, 
                            n_filters=32, 
                            kernel_sizes=[11, 21, 41],
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

class InceptionTimeClassifier(Generic_Model):

    def __init__(self, args, Data):
       
        self.Process_Args(args)
        self.Process_Data_To_Dataloaders(Data)
        self.Prep_Normalization(args, Data)
        # self.Prep_LossFunction(args, Data)  # Init CrossEntropy instead
        
        # Initialize loss function
        if 'y_train' in Data.keys():     
            self.Loss_Params = Get_Loss_Params(args, Train_Y = Data['y_train'])
        elif ('Loss_Type' in args.keys()):
            self.Loss_Params = Get_Loss_Params(args) 
        else:
            args['Loss_Type'] == 'CrossEntropyLoss'
            print ('Defaulting to CrossEntropyLoss')
            self.Loss_Params = Get_Loss_Params(args) 
        
        # Initialize a model here if 'x_train' was passed (otherwise do so in load())
        if 'x_train' in Data.keys():
            in_channels = Data['x_train'].shape[-1]
            self.model = InceptionTime(in_channels=in_channels)
            if 'y_train' in Data.keys():
               self.model[-1] = nn.Linear(in_features=4*32*1, out_features=len(np.unique(Data['y_train'])), bias=True)
            self.model.to(self.device)
            # Prep optimizer and scheduler
            self.Prep_Optimizer_And_Scheduler()
            
        # Initialize training performance
        self.Val_Best_Loss = 9999999
        self.Perf = []
        

# %%
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)
        self.Load_Random_State(Import_Dict)
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        self.Load_Normalization(Import_Dict)

        # initialize model, update model shape, send to GPU, update weights, load optimizer and scheduler
        
        # Find correct model sizes
        output_class_count = Import_Dict['model_state_dict']['4.weight'].shape[0]
        if ('0.inception_1.bottleneck.weight' in Import_Dict['model_state_dict'].keys()):
            input_chan_count = Import_Dict['model_state_dict']['0.inception_1.bottleneck.weight'].shape[1]
        else:
            input_chan_count = 1
            
        # Initialize model and load model
        self.model = InceptionTime(in_channels=input_chan_count)
        self.model[-1]  = nn.Linear(in_features = 4*32*1, out_features=output_class_count, bias=True)
        self.model.to(self.device)
        self.model.load_state_dict(Import_Dict['model_state_dict'])
        
        # Prep optimizer and scheduler
        self.Prep_Optimizer_And_Scheduler()
        
        
        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])

    
    # %%Overwrite how we adjust _multiple_ inputs. InceptionTime wants a different shape for data (not NCHW but NHW)
    def Adjust_Many_Images(self, image_batch):
        # This function is called after the image_batch is sent to GPU
        image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
        return image_batch
    