# -*- coding: utf-8 -*-
"""
Follows https://www.researchgate.net/publication/334407179_ECG_Arrhythmia_Classification_Using_STFT-Based_Spectrogram_and_Convolutional_Neural_Network 
per reviewer 3's suggestions

"""
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

class Spect_CNN(nn.Module):
    # just wrapping the LSTM here to return the correct h/c outputs, not the outputs per ECG time point
    # https://discuss.pytorch.org/t/cnn-lstm-for-video-classification/185303/7
    
    def __init__ (self):
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
            nn.Linear(in_features = 246016, out_features = 2)
            )
        
        self.Spect = torchaudio.transforms.MelSpectrogram(sample_rate = 400, n_mels = 512, n_fft=1024, hop_length=8).to('cuda')
        

    def forward(self, input_ecg):
        a = self.Spect( torch.transpose( input_ecg[:,0,:,:],2,1)) # NCHW 32 x 12 x 512 x 513
        ret = self.model(a[:,:,:,:512]) # cut last freq to line up size
        return ret # output is N x output_shape


class SpectCNNClass(GenericModel):
    
    def Gen_New_Model(self):
        self.model = Spect_CNN()
    
    def __init__(self, args, Data):
        
        #1. Process_Args
        self.Process_Args(args)
        
        #2. Prep_Normalization
        self.prep_normalization_and_reshape_data(args, Data)
        
        #3. Prep_Dataloaders_and_Normalize_Data
        self.Prep_Dataloaders_and_Normalize_Data()
        
        # 4. Prep Loss Params
        # Prep loss function w crossentropy default
        self.Prep_LossFunction(args, Data) # Classifier default is crossentropy
        
        a = time.time()
        # 5. Build your model
        self.Gen_New_Model()
        self.model.to(self.device)
            
        # 6. Prep optimizer and scheduler
        # self.Try_LSTM_Wrap()
        self.Prep_Optimizer_And_Scheduler()
        
        print('FFClass: Model/Optimizer/Scheduler T= ',time.time()-a)
        
        
    # %% Load
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        # self.Load_Normalization(Import_Dict) # we frontload normalization, so this no longer matters
        
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
            print('optimizer loaded')
        else:
            print('NO optimizer loaded')
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
            print('scheduler loaded')
        else:
            print("NO scheduler loaded")
            
        # new_params = self.optimizer.param_groups # compare these
        # breakpoint() # trying next lines
        self.Load_Random_State(Import_Dict)
    
    # %%Overwrite image adjustment - Ribeiro wants a different shape for data
    def Adjust_Many_Images(self, image_batch):
        # This function is called after the image_batch is sent to GPU
        # image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
        return image_batch