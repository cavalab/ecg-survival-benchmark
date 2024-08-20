# Ribeiro_Classifier is a Generic_Model
# args['Norm_Func'] defaults to 'None'
# args['seq_length'] defaults to 4096
# args['sample_freq'] defaults to 400Hz
# args['scale_multiplier'] defaults to 10 (all inputs are scaled by 10)
# args['dropout_rate'] defaults to 0.8
# args['kernel_size'] defaults to 17

# Originally from https://github.com/antonior92/ecg-age-prediction
# Heavily modified by PVL to line up with rest of flow
# Where possible, the original code was kept

import json
import torch
import os
from tqdm import tqdm

# import One_D_Resnet_Support
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

from MODELS.GenericModel import  GenericModel

# import Support_Functions
from MODELS.Support_Functions import Custom_Dataset
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss



class Ribeiro_Classifier(GenericModel):
    
    def __init__(self, args, Data):
        
        print('Ribeiro Classifier currently expects signals to have length 4096')
        
        #1. Process_Args
        self.Process_Args(args)
        
        #2. Prep_Normalization
        # Ribeiro model defaults to no normalization of input
        if ('Norm_Func' not in args.keys()):
            args['Norm_Func'] = 'None'
            print('By default, Ribeiro model does not normalize inputs')
        self.prep_normalization_and_reshape_data(args, Data)
        
        #3. Prep_Dataloaders
        self.Prep_Dataloaders_and_Normalize_Data()
        
        # 4. Prep Loss Params
        # Prep loss function w crossentropy default
        # self.Prep_LossFunction(args, Data) # Classifier default is crossentropy
        if 'y_train' in Data.keys():     
            self.Loss_Params = Get_Loss_Params(args, Train_Y = Data['y_train'])
        elif ('Loss_Type' in args.keys()):
            self.Loss_Params = Get_Loss_Params(args) 
        else:
            args['Loss_Type'] == 'CrossEntropyLoss'
            print ('Defaulting to CrossEntropyLoss')
            self.Loss_Params = Get_Loss_Params(args) 
            
        a = time.time()
        # 5. Build your model
        # Extra arg processing
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
            
        # Prep a model, send to device
        if 'y_train' in Data.keys():
            N_LEADS = Data['x_train'].shape[3]
        else:
            N_LEADS = 12 # the 12 leads
       
        
        self.single_model_init()
        self.model.to(self.device)
            
        # 6. Prep optimizer and scheduler
        # self.Try_LSTM_Wrap()
        self.Prep_Optimizer_And_Scheduler()
        
        print('RibeiroClass: Model/Optimizer/Scheduler T= ',time.time()-a)
        
    def single_model_init(self):
        if (self.LSTM == False):
            N_CLASSES = 2  
        else:
            N_CLASSES = self.LSTM_Feat_Size
            
        N_LEADS = 12
        self.model = ResNet1d(input_dim=(N_LEADS, self.seq_length),
                         blocks_dim=list(zip(self.net_filter_size, self.net_seq_lengh)),
                         n_classes=N_CLASSES,
                         kernel_size=self.kernel_size,
                         dropout_rate=self.dropout_rate)
        
        
        
        
        
    # %% Load
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)   
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        # self.Load_Normalization(Import_Dict) # we're frontloading normalization, so that doesn't matter

        # self.single_model_init()
        # self.Try_LSTM_Wrap() # have LSTM wrap this model
        # self.model.to(self.device) #
        
        # self.Prep_Optimizer_And_Scheduler() # re-build and re-attach optimizer to model
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
        
        
    
    # %%Overwrite image adjustment - Ribeiro wants a different shape for data
    def Adjust_Many_Images(self, image_batch):
        # This function is called after the image_batch is sent to GPU
        image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
        return image_batch