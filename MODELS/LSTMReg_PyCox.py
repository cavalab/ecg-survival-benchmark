# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:00:12 2024

@author: CH242985
"""
import torch
import os
from tqdm import tqdm


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
from MODELS.Generic_Model_PyCox import Generic_Model_PyCox

# import Support_Functions
from MODELS.Support_Functions import Custom_Dataset
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args
from MODELS.Support_Functions import Structure_Data_NCHW
from MODELS.Support_Functions import Get_Norm_Func_Params
from MODELS.Support_Functions import Normalize
from MODELS.Support_Functions import Get_Loss_Params
from MODELS.Support_Functions import Get_Loss

class CustomLSTM(nn.Module):
    # just wrapping the LSTM here to return the correct h/c outputs, not the outputs per ECG time point
    # https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7
    
    def __init__ (self, hidden_size = 1, num_layers = 1, output_size = 1):
        super(CustomLSTM, self).__init__()
        
        self.LSTM = nn.LSTM(input_size = 12, hidden_size = hidden_size, num_layers = num_layers, batch_first=True) # tte/e
        self.ff = nn.Linear(in_features = hidden_size, out_features = output_size) # 2 classes for crossentropy loss on a classifier
        

    def forward(self, input_ecg):
        # Expect input ECG to be N x 4096 samples x 12 channels
        
        # breakpoint()
        
        self.LSTM.flatten_parameters()
        out, (h, c) = self.LSTM(input_ecg, None) # c - num layers
        # output is the output after each input with shape [#ecg sequences in batch] x [length of longest sequence] x 1
        # h is hidden state at end of sequence. h[-1] is the output of the last layer
        # c is cell state (long term memory) at end of sequence
        # the output is the set of hidden states along the way.
        
        ret = self.ff(h[-1]) # run a linear layer at the end to compress hidden dimension to module output dimension
        
        return ret # output is N x output_shape

class LSTMReg_PyCox(Generic_Model_PyCox):
    
    def Gen_New_Model(self):
        
        # 540k params (12,000 in FF)
        self.model = CustomLSTM(hidden_size = 120, num_layers = 5, output_size = self.Num_Classes) # PyCox regression case - for CoxPH output has dim 1, else time_segments (usually 100)
            
    
    def __init__(self, args, Data):
        # Generic PyCox init
        #1. Process_Args
        self.Process_Args_PyCox(args)
        
        #2. Prep_Normalization
        self.Prep_Data_Normalization_Discretization(args, Data)
        
        #3. Prep_Dataloaders_and_Normalize_Data
        self.Process_Data_To_Dataloaders()
        
        # 4. Prep Loss Params
        # Prep loss function w crossentropy default
        # self.Prep_LossFunction(args, Data) # Classifier default is crossentropy
        
        a = time.time()
        # 5. Build your model
        self.Gen_New_Model()
        self.model.to(self.device)
            
        # 6. Prep optimizer and scheduler
        # self.Try_LSTM_Wrap()
        # self.Prep_Optimizer_And_Scheduler()
        
        self.pycox_mdl = self.Get_PyCox_Model() # init the optimizer and scheduler
        
        print('LSTMType: Init Model/Optimizer/Scheduler T= ',time.time()-a)
        
        
    # %% Load
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        # self.Load_Normalization(Import_Dict) # we frontload normalization based on Train data, so this no longer matters
        
        # print("ERROR: Should not be normalizing data in Load")
        # self.Prep_Dataloaders_and_Normalize_Data()

        # If you initialize a new model, you DO need to tell the optimizer and scheduler.
        # self.Gen_New_Model()
        # self.model.to(self.device)
        # self.Try_LSTM_Wrap() # convert to LSTM
        # self.Prep_Optimizer_And_Scheduler() # re-build and re-attach optimizer to model
        
        
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
            
        # new_params = self.optimizer.param_groups # compare these
        # breakpoint() # trying next lines
        self.Load_Random_State(Import_Dict)
    
    # %%Overwrite image adjustment - Ribeiro wants a different shape for data
    def Adjust_Image(self, single_image):
        single_image = single_image[0,:,:]
        return single_image # Just chan x leng, so 12 x 4k
    
    def Adjust_Many_Images(self, image_batch):
        # This function is called after the image_batch is sent to GPU
        # image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
        # image_batch = image_batch[:,0,:,:] # N-Len-Width
        return image_batch