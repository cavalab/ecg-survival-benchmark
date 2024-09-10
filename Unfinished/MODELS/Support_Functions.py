# -*- coding: utf-8 -*-
"""
This includes several functions that may overlap between models

Many implementations are from online tutorials / stackexchange
"""

import torch
import torch.nn as nn
import numpy as np

# for brier and concordance evaluations
import pandas as pd # monkeypatch
pd.Series.is_monotonic = pd.Series.is_monotonic_increasing



import os, h5py

from torch.utils.data.sampler import BatchSampler


# %% Generic Dataset used by classifiers

    

    
# %% Save out a neural net
def Save_NN(epoch, model, path, best_performance_measure=9999999, optimizer=None, scheduler=None, NT=None, NM=None, NS=None):
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    # best_performance_measure refers to the performance of the best model so far
    # so we don't accidentally overwrite it

    Out_Dict = {}
    Out_Dict['epoch'] = epoch
    Out_Dict['model_state_dict'] = model.state_dict()
    Out_Dict['Numpy_Random_State'] = np.random.get_state()
    Out_Dict['Torch_Random_State'] = torch.get_rng_state()
    Out_Dict['CUDA_Random_State'] = torch.cuda.get_rng_state()
    Out_Dict['best_performance_measure'] = best_performance_measure
    
    if (optimizer is not None):
        Out_Dict['optimizer_state_dict'] = optimizer.state_dict()
    if (scheduler is not None):
        Out_Dict['scheduler_state_dict'] = scheduler.state_dict()
        
    # Normalization Parameters
    if (NT is not None):
        Out_Dict['NT'] = NT # normalization type
        
    if (NM is not None):
        Out_Dict['NM'] = NM # normaliation mean per channel
        
    if (NS is not None):
        Out_Dict['NS'] = NS # normalization stdev per channel
    torch.save(Out_Dict, path)
    
# %% Save training args to json
def Save_Train_Args(path, train_args):
    import json
    with open(path, 'w') as file:
         file.write(json.dumps(train_args)) # use `json.loads` to do the reverse

# %% Split indices of Y into Tr/Val/Test based on ratios
def Data_Split_Rand( Y, Tr, V, Te):
    from torch.utils.data import random_split
    # 1. Separate X and Y. Ratio is given by Tr, V, Te
    # 2. Track number of Tr, V, Te, per class [for weighted sampling later]
    # 3. (12/4/2023) prioritize training count, then test count, then validation (because sometimes you get just one training sample...)
    
    Train_Inds = []
    Val_Inds = []
    Test_Inds = []

    Tr_Count = int(np.ceil(len(Y) * Tr / (Tr + V + Te)))
    Te_Count = int(np.ceil( (len(Y) - Tr_Count) * Te / (V + Te)))
    V_Count = len(Y) - Tr_Count - Te_Count
    
    Tr_Ind, V_Ind, Te_Ind = random_split(range(len(Y)), [Tr_Count, V_Count, Te_Count])
    
    Train_Inds = Train_Inds + list(Tr_Ind)
    Val_Inds = Val_Inds + list(V_Ind)
    Test_Inds = Test_Inds + list(Te_Ind)
      
    return Train_Inds, Val_Inds, Test_Inds


# %% Standardize input shape
def Structure_Data_NCHW(arr):
    # We receive a numpy array that is either 2D (N-H)
    # Or 3D (N-H-W)
    # or 4D (N-C-H-W)
    # and we want to expand dims until we get the 4D version
    if (len(arr.shape)==2): # N-H to N-H-W
        arr = np.expand_dims(arr,axis=-1)
        
    if (len(arr.shape)==3): # N-H-W to N-C-H-W
        arr = np.expand_dims(arr,axis=1)
        
    return arr


# %% Normalization
def Get_Norm_Func_Params (args, Some_Array): # figure out how to normalize (from train set)
    if ('Norm_Func' not in args.keys()):
        args['Norm_Func'] = 'nchW'
        print('By default, using nchW normalization')

    if (args['Norm_Func'] == 'None'):        
        norm_type = 'No_Norm'
        u = 0
        s = 0
        
    if (args['Norm_Func'] == 'nChw'):
        # Get u, s per C
        norm_type = 'nChw'
        u = [ np.mean(Some_Array[:,k,:,:]) for k in range(Some_Array.shape[1])]
        s = [ np.std(Some_Array[:,k,:,:])  for k in range(Some_Array.shape[1])]
        
    if (args['Norm_Func'] == 'nchW'):
        # Get u, s per W
        norm_type = 'nchW'
        u = [np.mean(Some_Array[:,:,:,k]) for k in range(Some_Array.shape[3])]
        s = [np.std(Some_Array[:,:,:,k])  for k in range(Some_Array.shape[3])]
    
    return norm_type, u, s # these can be saved

     


def Normalize (value, norm_type, u, s): # apply normalization to a batch input (like a test set)
    # input: batch NCHW pytorch tensor [passed by ref]
    # output: modified cloned output
    
    
    if norm_type == 'No_Norm':
        asdf = value
    
    if norm_type =='nChw':
        asdf = torch.stack([ (value[:,k,:,:] - u[k]) / s[k] for k in range(len(u))],dim=1)
    
    if norm_type =='nchW':
        # asdf = torch.stack([ (value[:,:,:,k] - u[k]) / s[k] for k in range(len(u))],dim=3) # RAM hungry, don't use
        for k in range(len(u)):
            value[:,:,:,k] = (value[:,:,:,k] - u[k]) / s[k]
        asdf = value
        
    return asdf


# %% Loss Functions
def Get_Loss_Params (args, Train_Y=None): # figure out how to apply losses
    Loss_Params = {}
    if ('Loss_Type' in args.keys()):
        Loss_Params['Type'] = args['Loss_Type']
    else:
        Loss_Params['Type'] = 'None'
        print('Loss_Type not in args - exiting.')
        quit()
        
    # if loss needs weights of 1/count, get weights. Modified from Ribeiro.
    if ( (Loss_Params['Type'] == 'wSSE') or (Loss_Params['Type']=='wSAE') ):
        unique_vals, counts = np.unique(Train_Y, return_counts=True)
        weights = 1 / counts
        Val_Weight_Map = {int(unique_vals[k]):weights[k] for k in range(len(weights))}
        Loss_Params['Weight_Map'] = Val_Weight_Map
    return Loss_Params
        

def Get_Loss(Model_Out, Correct_Out, Loss_Params): # Get a loss for a batch

    # Classifier losses
    if (Loss_Params['Type'] == 'CrossEntropyLoss'):
        temp = nn.CrossEntropyLoss() # takes like 1ms
        Correct_Out = Correct_Out.type(torch.LongTensor) #  LongTensor has to be set before going to GPU
        Correct_Out = Correct_Out.to(Model_Out.device)
        return temp(Model_Out, Correct_Out)

    # Regression Losses 
    Correct_Out = torch.unsqueeze(Correct_Out,dim=1)
    
    if (Loss_Params['Type'] == 'SSE'):
        return sum(  (Model_Out - Correct_Out)**2  )

    if (Loss_Params['Type'] == 'wSSE'):
        temp = (Model_Out - Correct_Out)**2 
        a = [Loss_Params['Weight_Map'][k.item()] for k in Correct_Out]
        weights = torch.from_numpy( np.array( a ) ).to(Correct_Out.device)
        weights = weights.unsqueeze(dim=1)
        temp_scaled = torch.mul(temp, weights)
        return sum(temp_scaled)
    
    if (Loss_Params['Type'] == 'SAE'):
        return sum(  abs(Model_Out - Correct_Out)  )
 
    if (Loss_Params['Type'] == 'wSAE'):
        temp = abs(Model_Out - Correct_Out)
        a = [Loss_Params['Weight_Map'][k.item()] for k in Correct_Out]
        weights = torch.from_numpy( np.array( a ) ).to(Correct_Out.device)
        weights = weights.unsqueeze(dim=1)
        temp_scaled = torch.mul(temp, weights)
        return sum(temp_scaled)

    return 0


    
# %% Utility: append variable to existing hdf5 file
def Save_to_hdf5(path, var, var_name):
    # if hdf5 file DNE, make it
    # add variable that file, overwriting past entries
    if (os.path.isfile(path) == False):
        with h5py.File(path, "w") as f:
            f.create_dataset(var_name, data = var)
            print('saved ' + var_name)
    else:
        with h5py.File(path, "r+") as f:
            database_list = [k for k in f.keys()]
            if var_name in database_list:
                # updated 08/22/24
                del f[var_name]
                f.create_dataset(var_name, data = var)
                print('updated ' + var_name)
            else:
                f.create_dataset(var_name, data = var)
                print('saved ' + var_name)

# %%              
# def debug_load():
#     old_model = copy.deepcopy(self.model)
#     old_np_state = np.random.get_state()
#     old_torch_state = torch.get_rng_state()
    
#     # from https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
#     for p1, p2 in zip(old_model.parameters(), self.model.parameters()):
#         if p1.data.ne(p2.data).sum() > 0:
#             print ('parameter mismatch')
    
#     a = old_np_state[1]
#     b = np.random.get_state()[1]
#     if (sum(a!=b) > 0):
#         print('NP rand err')
    
#     if (torch.equal(old_torch_state, torch.get_rng_state()) == False): # random state is different
#         print('Torch rand err')    
    
#  %% Interrogate hdf5
# C:\Users\ch242985\Desktop\ECG\CNN\Trained_Models\MIMICIV_Multimodal_Subset\TimesNetClass_1\EVAL\MIMICIV_Multimodal_Subset Test_Folder

# import h5py
# import os
# with h5py.File('Stored_Model_Output.hdf5', "r") as f:
#     print(f.keys())
#     for key in f.keys():
#         print(f[key][()].shape)
        