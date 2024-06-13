# Ribeiro_Regression_PyCox is a Generic_Model_PyCox, which in turn is a Generic_Model
# args['Norm_Func'] defaults to 'None'
# args['seq_length'] defaults to 4096
# args['sample_freq'] defaults to 400Hz
# args['scale_multiplier'] defaults to 10 (all inputs are scaled by 10)
# args['dropout_rate'] defaults to 0.8
# args['kernel_size'] defaults to 17

# Originally from https://github.com/antonior92/ecg-age-prediction
# Heavily modified by PVL to line up with the rest of the survival modeling flow
# Where possible, the original code was kept


import json
import torch
import os
from tqdm import tqdm

from MODELS.Ribeiro_Support import ResNet1d

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


class Ribeiro_Regression_PyCox(Generic_Model_PyCox):
    
    def __init__(self, args, Data):
        # Generic_Model_PyCox init: 1) Process Args 
        self.Process_Args_PyCox(args) # process common arguments
        
        # Ribeiro model defaults to no normalization of input
        if ('Norm_Func' not in args.keys()):
            args['Norm_Func'] = 'None'
            print('By default, Ribeiro model does not normalize inputs')
        
        # Extra processing on args
        if ('seq_length' in args.keys()):
            self.seq_length = int(args['seq_length'])
        else:
            self.seq_length = 4096
            
        if ('sample_freq' in args.keys()):
            self.sample_freq = int(args['sample_freq'])
        else:
            self.sample_freq = 400
            
        if ('scale_multiplier' in args.keys()): # help='multiplicative factor used to rescale inputs.')
            self.scale_multiplier = int(args['scale_multiplier'])
        else:
            self.scale_multiplier = 10

        # ... here we're going to hard-code net filter size, cause we can't get that from a set of strings
        print('Ribeiro Models currently expect signals to have length 4096')
        self.net_filter_size = [64, 128, 196, 256, 320] #'filter size in resnet layers (default: [64, 128, 196, 256, 320]).'
        self.net_seq_lengh = [4096, 1024, 256, 64, 16] #'number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).'
            
        if ('dropout_rate' in args.keys()): #help='reducing factor for the lr in a plateu (default: 0.1)')
            self.dropout_rate = float(args['dropout_rate'])
        else:
            self.dropout_rate = 0.8
        
        if ('kernel_size' in args.keys()):
            self.kernel_size = int(args['kernel_size'])
        else:
            self.kernel_size = 17
        self.args['kernel_size'] = self.kernel_size
            
        # Generic_Model_PyCox init: 2) Prep Normalization
        self.Prep_Data_Normalization_Discretization(args, Data) # sets num_durations, frontloads normalization and discretization, self.Data = Data, if (xtrain)
        
        # Generic_Model_PyCox init: 3) Prep DataLoaders
        if (self.max_duration is not None):
            self.Process_Data_To_Dataloaders()
            
        # Generic_Model_PyCox init: 4) Prep a model, send to device
        if 'y_train' in Data.keys():
            N_LEADS = Data['x_train'].shape[3]
        else:
            N_LEADS = 12 # the 12 leads

        self.model = ResNet1d(input_dim=(N_LEADS, self.seq_length),
                         blocks_dim=list(zip(self.net_filter_size, self.net_seq_lengh)),
                         n_classes=self.Num_Classes,
                         kernel_size=self.kernel_size,
                         dropout_rate=self.dropout_rate)
        self.model.to(self.device)
        self.pycox_mdl = self.Get_PyCox_Model() # init the optimizer and scheduler
        

        # Generic_Model_PyCox init: 5) Prep validation loss and performance storage list
        self.Val_Best_Loss = 9999999
        self.Perf = []
        
    # %% Load
    def Load(self, best_or_last):
        
        # Load random state (I still don't know why train 2 epoch ~ = train 1, load, train 1)
        Import_Dict = self.Load_Checkpoint(best_or_last)
        self.Load_Random_State(Import_Dict)
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        self.Load_Normalization(Import_Dict)

        # initialize model, update model shape, send to GPU, update weights, load optimizer and scheduler
        N_LEADS = 12
        self.Num_Classes = Import_Dict['model_state_dict']['lin.weight'].shape[0] # number of output classes
        self.model = ResNet1d(input_dim=(N_LEADS, self.seq_length),
                         blocks_dim=list(zip(self.net_filter_size, self.net_seq_lengh)),
                         n_classes=self.Num_Classes,
                         kernel_size=self.kernel_size,
                         dropout_rate=self.dropout_rate)
        self.model.to(self.device)
        self.model.load_state_dict(Import_Dict['model_state_dict'])
        
        self.pycox_mdl = self.Get_PyCox_Model() # also inits the optimizer and scheduler

        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])

        # Discretization: Load and Set Up
        self.Discretize_On_Load(Import_Dict)
            

    
    # %% Overwrite how we adjust _each_ input and _multiple_ inputs. Ribeiro wants a different shape for data (not NCHW but NHW)
    def Adjust_Image(self, single_image):
        single_image = torch.transpose(single_image, 1,2)
        return single_image[0] # Just chan x leng, so 12 x 4k
    
    def Adjust_Many_Images(self, many_images):
        many_images = torch.transpose(many_images, 1,2)
        return many_images # Just chan x leng, so 12 x 4k