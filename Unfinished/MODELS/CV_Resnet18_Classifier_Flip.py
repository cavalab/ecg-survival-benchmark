# Unlike Resnet18_Classifier, this one looks at the 12 channels as colors

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
class CV_Resnet18_Classifier_Flip(GenericModel):

    def __init__(self, args, Data):
        
        self.Process_Args(args)
        self.Process_Data_To_Dataloaders(Data)
        self.Prep_Normalization(args, Data)
        
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
        
        # Prep a default model
        self.model = models.resnet18(weights = None)
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias=False)
        # Set last layer size based on number of classes in training data, if y_train is available
        if 'y_train' in Data.keys():
            self.model.fc = nn.Linear(in_features=512, out_features=len(np.unique(Data['y_train'])), bias=True)
        self.model.to(self.device)
        
        # Prep optimizer and scheduler
        self.Prep_Optimizer_And_Scheduler()
        
        # Prep to train - performance values can be overwritten by load
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
        Actual_output_class_count = Import_Dict['model_state_dict']['fc.weight'].shape[0]
        self.model.fc = nn.Linear(in_features=512, out_features=Actual_output_class_count, bias=True)
        self.model.to(self.device)
        self.model.load_state_dict(Import_Dict['model_state_dict'])
        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
            
        
    # %%Overwrite image adjustment - Inceptiontime wants a different shape for data
    def Adjust_Many_Images(self, image_batch):
        # breakpoint()
        image_batch = torch.transpose(image_batch,1,3)
        # This function is called after the image_batch is sent to GPU
        # image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
        return image_batch