# InceptionTImeClassifier is a Generic_Model
# args['Loss Type'] defaults to CrossEntropyLoss
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

class InceptionTimeClassifier(GenericModel):

    def __init__(self, args, Data):
       
        self.Process_Args(args)
        self.prep_normalization_and_reshape_data(args, Data)
        self.Prep_Dataloaders_and_Normalize_Data()
        
        # self.Prep_LossFunction(args, Data)  # Init CrossEntropy instead
        # Initialize loss function
        if 'y_train' in Data.keys():     
            self.Loss_Params = Get_Loss_Params(args, Train_Y = Data['y_train'])
        elif ('Loss_Type' in args.keys()):
            self.Loss_Params = Get_Loss_Params(args) 
        else:
            args['Loss_Type'] == 'CrossEntropyLoss'
            print ('InceptionTimeClassifier: Defaulting to CrossEntropyLoss')
            self.Loss_Params = Get_Loss_Params(args) 
        
        # Initialize a model
        a = time.time()
        # if 'x_train' in Data.keys():
        #     in_channels = Data['x_train'].shape[-1]
        in_channels=12
        self.model = InceptionTime(in_channels=in_channels)
        
        if (self.LSTM == False):
            self.model[-1] = nn.Linear(in_features=4*32*1, out_features=2, bias=True)
        else:
            self.model[-1] = nn.Linear(in_features=4*32*1, out_features=self.LSTM_Feat_Size, bias=True)
            
        self.model.to(self.device)
        
        # self.Try_LSTM_Wrap()
        # Prep optimizer and scheduler
        self.Prep_Optimizer_And_Scheduler()
        print('InceptionClass: Model/Optimizer/Scheduler T= ',time.time()-a)
        

# %%
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        # self.Load_Normalization(Import_Dict) # we're frontloading normalization, so this no longer matters
        
        # print("ERROR: Should not be normalizing data in Load")
        # self.Prep_Dataloaders_and_Normalize_Data()
        

        # initialize model, update model shape, send to GPU, update weights, load optimizer and scheduler
        
        # Find correct model sizes
        # if ('LSTM' in self.args.keys() and self.args['LSTM']=='True'):
        #     output_class_count = Import_Dict['model_state_dict']['CNN.4.weight'].shape[0]
        # else:
        #     output_class_count = Import_Dict['model_state_dict']['4.weight'].shape[0]
        
        # output_class_count = 1

        # if ('0.inception_1.bottleneck.weight' in Import_Dict['model_state_dict'].keys()):
        #     input_chan_count = Import_Dict['model_state_dict']['0.inception_1.bottleneck.weight'].shape[1]
        # else:
        #     input_chan_count = 1
            
        # # Initialize model and load model
        # self.model = InceptionTime(in_channels=input_chan_count)
        # self.model[-1]  = nn.Linear(in_features = 4*32*1, out_features=output_class_count, bias=True)
        # self.model.to(self.device)
        
        # # lstm compat: after building model, re-prep optimizer and scheduler so field names line up
        # if ('LSTM' in self.args.keys() and self.args['LSTM']=='True'):
        #     self.Prep_Optimizer_And_Scheduler()
            
        self.model.load_state_dict(Import_Dict['model_state_dict'])
        
        # Prep optimizer and scheduler
        # self.Prep_Optimizer_And_Scheduler()
        
        
        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
            
        self.Load_Random_State(Import_Dict)

    
    # %%Overwrite how we adjust _multiple_ inputs. InceptionTime wants a different shape for data (not NCHW but NHW)
    def Adjust_Many_Images(self, image_batch):
        # This function is called after the image_batch is sent to GPU
        image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
        return image_batch
    