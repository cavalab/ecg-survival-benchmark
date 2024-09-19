"""
MODEL_RUNNER_PyCox 

Overall:
    Process arguments
    Load training data
    Initialize a model
    (optional) load model parameters
    Train a model (can be skipped with --epoch_end -1)
    Evaluate a model
    Save model performance
    
Essential Arguments (no defaults):
    --Model_Name [ModelType + '_' + Name]       ex: --Model_Name RibeiroClass_Bob
    --Train_Folder [FolderName]                 ex: --Train_Folder Code15
    --Test_Folder [FolderName]                  ex: --Test_Folder Code15
    --Eval_Dataloader ['Validation', 'Train', or 'Test']   ex: --Eval_Dataloader Test

Optional Arguments:
    --Load ['Best' or 'Last'] - loads a model before training   ex: --Load Best
    --batch_size [int]                                     ex: --batch_size 128
    --validate_every [int]                                 ex: --validate_every 10
    --epoch_end [int]                                      ex: --epoch_end 25
       note: epoch_end -1 (default) skips training
    
---
Model_Runner:
    - Model_Runner Interprets passed arguments
        - Either from Command Line (main)
        - Or from another script (Run_Model_via_String_Arr)
            ex: args = '--Train_Folder '+folder+' --Model_Name Resnet18Class_asfd1 --Train True --Test_Folder '+folder+' --batch_size 128 --epoch_end 10 --validate_every 10'

    - Quick note on Models:
        - Model Type is set by the first half (up to an underscore) of their name: --Model_Name Resnet18Class_[asfd1]
        - Trained models live in Trained_Models/[their training data folder], as passed in: --Train_Folder [folder]
        - Model_Runner MUST get the Train_Folder and Model_Name arguments to know which model to build or use
        
    - Model_Runner:
        - Sets Model Name 
        - Determines if training happens (requires --Train True AND --Train_Folder [folder])
        - Sets a random seed if that was not passed
        - If training, Loads training data. 
            - Data is currently (12/4/23) assumed to be in:
                - In os.path.join(os.getcwd(),'HDF5_DATA',args['Train_Folder'],'TRAIN_DATA','Train_Data.hdf5')
                - With numpy parameters 'x' and 'y'
        - If evaluating: (requires --Test_Folder [folder])
            - Attempts to load from os.path.join(os.getcwd(),'HDF5_DATA',args['Test_Folder'],'TEST_DATA','Test_Data.hdf5')
            - On failure:
                - Checks for a passed TE_RATIO argument. By default, TR_RATIO = 80, VA_RATIO = 20.
                    - generates a test set from a training/validation/test split of:
                    - os.path.join(os.getcwd(),args['Test_Folder'],'TRAIN_DATA','Train_Data.hdf5')
        - If a training set is present:
            - Data is split randomly into Train/Test/Validate from TR_RATIO/TE_RATIO/VA_RATIO
            - Splits prioritize allocations for Train/Test/Validate in that order (ceil, ceil, remainder) (If a class only has one sample, it is in training.)
            - For classifiers, the split is performed per class
            - Everything is stored in Data['x_train'], Data['x_valid'], Data['x_test'], and y-equivalents
            
        - Initalizes a model: model(args, Data)
        - Runs model.load() if --Load [anything]
        - Runs model.eval() if --Test_Folder [anything]
        - Generates figures from eval, saves figure, eval params, and eval outputs
        
    - Other:
        - Model training and evaluation are split intentionally
        - Classifiers currently assume the lowest class number to be 0
        - If you specify a non-default (0 or 1) column of train_y or test_y to use, that will add sub-folders to the path
        

main() parses command line inputs, then feeds a dict to run_model()
Run_Model_via_String_Arr() parses arguments from an array of strings, then feeds a dict to run_model()

DATA is assumed to be in N-H or N-H-W or N-C-H-W format. 
For time series, this would be N-Length or N-Length-Chan or N-[Color]-Length-Chan
Currently, all models reshape it into N-C-H-W

"""

# handle pycox folder requirement FIRST
import os
os.environ['PYCOX_DATA_DIR'] = os.path.join(os.getcwd(),'Mandatory_PyCox_Dir')

from Model_Runner_Support import get_covariates

from Model_Runner_Support import Load_Data
from Model_Runner_Support import Clean_Data
from Model_Runner_Support import Augment_Y
from Model_Runner_Support import Split_Data
from Model_Runner_Support import DebugSubset_Data
from Model_Runner_Support import set_up_train_folders
from Model_Runner_Support import set_up_test_folders

# evaluation wrappers
from Model_Runner_Support import Gen_KM_Bootstraps
from Model_Runner_Support import Gen_Concordance_Brier_No_Bootstrap
from Model_Runner_Support import Gen_Concordance_Brier_PID_Bootstrap
from Model_Runner_Support import Gen_AUROC_AUPRC
from Model_Runner_Support import save_histogram

from MODELS.Support_Functions import Save_to_hdf5



from MODELS import Generic_Model_PyCox

import numpy as np
import torch
import matplotlib.pyplot as plt

import time

import collections
collections.Callable = collections.abc.Callable

import argparse



# %% 
def main(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args()
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
def Run_Model_via_String_Arr(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args(args[0])
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
def Run_Model(args):
    # input is paired dict of strings named args
    start_time = time.time()
    
    # %%1. CUDA check and arg processing
    for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)
       
    if (torch.cuda.is_available() == False):
        print('No CUDA. Exiting.')
        exit()
       
    # Grab model name. No point in proceeding without it.
    if ('Model_Name' not in args.keys()):
        print('Model_Name not specified - cant train or pull models')
        exit()
    Model_Type = args['Model_Name'].split('_')[0]
    args['Model_Type'] = Model_Type

    if ('Train_Folder' not in args.keys()):
        print('Train_Folder not specified - cant train or pull models')
        exit()
    
    # Set Random seeds - should really be from args. Note: "load" model will overwrite these!
    if ('Rand_Seed' in args.keys()):
        args['Rand_Seed'] = int(args['Rand_Seed'])
        
    if ('Rand_Seed' not in args.keys()):
        np.random.seed()
        args['Rand_Seed'] = np.random.randint(70000,80000)
        print('Rand Seed Not Set. Picking a random number 70,000 - 80,000... ' + str(args['Rand_Seed']))    
    
    np.random.seed(args['Rand_Seed'])
    torch.manual_seed(args['Rand_Seed'])
    torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
    
    # Y covariate indices get passed here
    val_covariate_col_list, test_covariate_col_list = get_covariates(args)
    
    # %% Process data: Load, Clean, Split
    Data, Train_Col_Names, Test_Col_Names = Load_Data(args) # Data is a dict, so passed by reference from now on
    Clean_Data(Data, args)       # remove TTE<0 and NaN ECG
    Augment_Y(Data, args)        # Augments Data['y_'] with [PID, TTE, E] for model train/runs without overwriting loaded information.
    Split_Data(Data)             # splits 'train' data 80/20 into train/val by PID
    DebugSubset_Data(Data, args) # If args['debug'] == True, limits Data[...] to 1k samples each of tr/val/test.
    
    # augment data with reshaped covariates
    for key in ['train', 'valid']:
        y_key = 'y_'+key
        z_key = 'z_'+key # for covariates
        Data[z_key] = Data[y_key][:,val_covariate_col_list]
        
    for key in ['test']:
        y_key = 'y_'+key
        z_key = 'z_'+key # for covariates
        Data[z_key] = Data[y_key][:,test_covariate_col_list]
    
    # %% 5. set up trained model folders if they  don't exist
    set_up_train_folders(args)
    
    # %% 6. Select model, (maybe) load an existing model. ask for training (runs eval after training reqs met)
    print('Model_Runner:  Got to model init. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    asdf = Generic_Model_PyCox.Generic_Model_PyCox(args, Data)
    
    # if (Model_Type == 'InceptionTimeReg'):
    #     asdf = InceptionTimeRegression_PyCox.InceptionTimeRegression_PyCox(args, Data)
    # if (Model_Type == 'RibeiroReg'):
    #     asdf = Ribeiro_Regression_PyCox.Ribeiro_Regression_PyCox(args, Data)
    # if (Model_Type == 'LSTMReg'):
    #     asdf = LSTMReg_PyCox.LSTMReg_PyCox(args, Data)
    # if (Model_Type == 'TimesNetReg'):
    #     asdf = TimesNetReg_PyCox.TimesNetReg_PyCox(args, Data)
    # if (Model_Type == 'SpectCNNReg'):
    #     asdf = SpectCNNReg_PyCox.SpectCNNReg_PyCox(args, Data)
      
    print('Model_Runner:  Got to Train. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    if( ('Load' in args.keys())):
        asdf.Load(args['Load'])
    asdf.train()
        
    
    # %% Generate and save out results
    
    # %% Model Evaluation
    # To evaluate model we need: dicrete time points, discretized time to event, event 0/1, S(t) per time point
    
    if ('Test_Folder' in args.keys()):
        
        print('Model_Runner:  got to model eval. Total Time elapsed: ' ,'{:.2f}'.format(time.time()-start_time))
        
        if ('num_durations' not in args.keys()):
            args['num_durations'] = '100'
            print('By default, using 100 time intervals')
            num_durations = 100
        else:
            num_durations = int(args['num_durations'])
        
        if ('Eval_Dataloader' in args.keys()):
            cuts, disc_y_t, disc_y_e, surv, surv_df = asdf.Test(Which_Dataloader = args['Eval_Dataloader']) # This lets you evaluate the model on its validation set instead of test set
        else:
            cuts, disc_y_t, disc_y_e, surv, surv_df = asdf.Test() 
            
        sample_time_points = cuts # where are we sampling the survival functions?
        
        # PyCox assumes test datasets aren't discretized, but we've discretized them, so adjust surv_df (affects concordance measures later)
        surv_df.index = np.arange(num_durations)
        
        
        # %% Set up folders and save eval args
        set_up_test_folders(args)
             
        # %% 15. Save out everything we need to recreate evaluation: 
        outputs_hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
        Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(outputs_hdf5_path, disc_y_t, 'disc_y_t') # when it really happened
        Save_to_hdf5(outputs_hdf5_path, disc_y_e, 'disc_y_e') # what really happened
        Save_to_hdf5(outputs_hdf5_path, Data['y_test'], 'y_test')
        Save_to_hdf5(outputs_hdf5_path, surv, 'surv')
        Save_to_hdf5(outputs_hdf5_path, Test_Col_Names + ['PID', 'TTE*', 'E*'], 'Test_Col_Names')


        # %% 16. evlauations
        
        # Save out KM. Add bootstrapping (20x). Saves KM values out separately in case you want to recreate that.
        Gen_KM_Bootstraps(surv, disc_y_e, disc_y_t, sample_time_points, args)

        # Concordance and Brier Score 
        time_points = [1,2,5,10,999]
        # across all ECG
        
        # Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e.astype(bool), time_points, sample_time_points, args)
        Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e, time_points, sample_time_points, args)
        # bootstrap: 1 ECG per patient x 20
        Gen_Concordance_Brier_PID_Bootstrap(Data, args, disc_y_t, disc_y_e, surv_df, sample_time_points, time_points)
        
        # AUROC and AUPRC
        time_points = [1,2,5,10] # 999 doesn't work for AUROC
        Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points, args)
        
        
        # histogram
        save_histogram(sample_time_points, disc_y_t, surv, args)
        
        print('Model_Runner: Finished evaluation. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
        
        
    #%% Test?
if __name__ == '__main__':
   main()