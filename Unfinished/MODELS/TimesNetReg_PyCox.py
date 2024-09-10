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


from MODELS.TimesNet_Support import Model as TimesNetModel

class Struct:
    # https://stackoverflow.com/questions/1305532/how-to-convert-a-nested-python-dict-to-object
    def __init__(self, **entries):
        self.__dict__.update(entries)

class TimesNetReg_PyCox(Generic_Model_PyCox):
    
    def Gen_New_Model(self):
        config_dict = {}
        config_dict['seq_len'] = 4096
        config_dict['dropout'] = 0 #? maybe?
        config_dict['label_len'] = 1
        config_dict['e_layers'] = 2
        config_dict['enc_in'] = 12 # I think?
        config_dict['d_model'] = 64 # they go up to 512 for larger cases
        config_dict['embed'] =  'fixed'#
        config_dict['freq'] = 400
        config_dict['num_class'] = self.Num_Classes # PyCox has 1...100 output channels
        
        config_dict['pred_len'] = 0 # don't predict
        config_dict['top_k'] = 3 # they use 3 for classification
        config_dict['d_ff'] = 64
        config_dict['num_kernels'] = 6
        
        configs = Struct(**config_dict)
        
        self.model = TimesNetModel(configs)
    
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
    def Adjust_Image(self, image_batch):
        return image_batch[0]
    
    def Adjust_Many_Images(self, image_batch):
        # This is ONLY called in PyCox CoxPH modeling, and is passed image_batch[:,0,:,:]. So don't  change anything.
        return image_batch