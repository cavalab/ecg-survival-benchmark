# args['horizon']           - float; {coreq} also requires args['y_col_train_time'] and args['y_col_train_event'], or args['y_col_test_time'] and args['y_col_test_event']. ignores args['y_col_train'] and args['y_col_test']
# args['y_col_train_time']  - int;   {coreq 'horizon'} 
# args['y_col_train_event'] - int;   {coreq^}
# args['y_col_test_time']   - int;   {coreq 'horizon'}
# args['y_col_test_event']  - int;   {coreq^}
# args['y_col_train']       - int; which column of Y is train data (ignored with horizon)
# args['y_col_test']        - int; which column of Y is test data  (ignored with horizon)

# New assumption 5/1/24: column 0 of Data[] is patient ID

"""
MODEL_RUNNER

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

[Not yet implemented] normalization is currently done per_color (N-_C_-H-W)
"""

# %% Imports; Support functions before main functions...

# handle pycox folder requirement FIRST
import os 
os.environ['PYCOX_DATA_DIR'] = os.path.join(os.getcwd(),'Mandatory_PyCox_Dir') # Fixes a bug on importing PyCox


# from Model_Runner_Support import get_covariates
from Model_Runner_Support import Load_Labels
from Model_Runner_Support import Apply_Horizon
from Model_Runner_Support import Split_Data
from Model_Runner_Support import Load_ECG_and_Cov

# from Model_Runner_Support import DebugSubset_Data
from Model_Runner_Support import set_up_train_folders
from Model_Runner_Support import set_up_test_folders

# evaluation wrappers
from Model_Runner_Support import Gen_KM_Bootstraps
from Model_Runner_Support import Gen_Concordance_Brier_No_Bootstrap
# from Model_Runner_Support import Gen_Concordance_Brier_PID_Bootstrap
from Model_Runner_Support import Gen_AUROC_AUPRC
from Model_Runner_Support import print_classifier_ROC
from Model_Runner_Support import save_histogram

from MODELS import GenericModelClassifierCox


from MODELS.Support_Functions import Save_to_hdf5


import pandas as pd
import matplotlib.pyplot as plt

from sksurv.linear_model import CoxPHSurvivalAnalysis
from scipy.special import softmax 

import numpy as np
import torch
import time
import json

import collections
collections.Callable = collections.abc.Callable

import argparse 

# to see how the classifier did at its job
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# %% Compatability - bring back older version of scipy simpson function
import scipy
from MODELS.Support_Functions import simps
scipy.integrate.simps = simps



# %% 
def main(*args):
    # convert passed args into a string-string dict so everything downstream can see it
    parser = argparse.ArgumentParser()
    _, unknown_args = parser.parse_known_args()
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
def Run_Model_via_String_Arr(*args):
    # convert passed args into a string-string dict so everything downstream can see it
    parser = argparse.ArgumentParser()
    _, unknown_args = parser.parse_known_args(args[0])
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
def Run_Model(args):
    # input is paired dict of strings named args
    start_time = time.time()
    
    # %% 1. CUDA check
    # CUDA
    for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)
       
    if (torch.cuda.is_available() == False):
        print('No CUDA. Exiting.')
        exit()
       
    # %% 2. Arg Processing
    
    assert('Model_Name' in args.keys())
    Model_Type = args['Model_Name'].split('_')[0]
    args['Model_Type'] = Model_Type

    assert('Train_Folder' in args.keys())
    
    # 3. Random Seeds should really be from args. Note: "load"ed models overwrite these!
    if ('Rand_Seed' in args.keys()):
        args['Rand_Seed'] = int(args['Rand_Seed'])
        
    if ('Rand_Seed' not in args.keys()):
        np.random.seed()
        args['Rand_Seed'] = np.random.randint(70000,80000)
        print('Rand Seed Not Set. Picking a random number 70,000 - 80,000... ' + str(args['Rand_Seed']))    
    
    np.random.seed(args['Rand_Seed'])
    torch.manual_seed(args['Rand_Seed'])
    torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
    
    if ('Eval_Dataloader' not in args.keys()): # This lets you evaluate the model on its validation set instead of test set
        args['Eval_Dataloader'] = 'Test'
    
    # %% Process data: Load, Clean, Split
    train_df, test_df = Load_Labels(args)       # Data is a dict, is passed by reference
    # Clean_Data(Data, args)       # remove TTE<0 and NaN ECG
    Apply_Horizon(train_df, test_df, args)    # classifiers need to compact TTE and E into a single value, E*. Augments Data['y_'] for model train/runs without overwriting loaded information.
    train_df, valid_df = Split_Data(train_df)             # splits 'train' data 80/20 into train/val by PID
    Data, train_df, valid_df, test_df = Load_ECG_and_Cov(train_df, valid_df, test_df, args)
    # DebugSubset_Data(Data, train_df, test_df, args) # If args['debug'] == True, limits Data[...] to 1k samples each of tr/val/test.
    
    # %% 9. set up model folders if they  don't exist
    set_up_train_folders(args)

    # %% 10. Select model, (maybe) load an existing model. ask for training (runs eval after training reqs met)
    print('Model_Runner: Got to init. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    
    my_model = GenericModelClassifierCox.GenericModelClassifierCox(args, Data, train_df, valid_df, test_df)
    
    print('Model_Runner:  Got to Train. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    if( ('Load' in args.keys())):
        my_model.Load(args['Load'])
    my_model.train()
        
    # %% 11. Generate and save out results
    print('got to eval. Total Time elapsed: ' ,'{:.2f}'.format(time.time()-start_time))
    if ('Test_Folder' in args.keys()):
        
        # 11.1 get model outputs for test, validation sets (unshuffled)
        test_outputs, test_Loss, test_correct_outputs = my_model.Test(Which_Dataloader = args['Eval_Dataloader'])
        val_outputs, val_Loss, val_correct_outputs    = my_model.Test(Which_Dataloader = 'Validation')
        
        # adjust output formats
        val_outputs  = np.squeeze(val_outputs)
        test_outputs = np.squeeze(test_outputs)
        
        # softmax the outputs
        val_outputs = np.array([softmax(k)[1] for k in val_outputs])
        test_outputs = np.array([softmax(k)[1] for k in test_outputs])
        
        # We have outputs - set up Folders
        set_up_test_folders(args)
        
        # Save out * for recreating Cox models later (e.g. subgroup analysis)
        # Save out smx val/test model outputs + the labels ( [PID, TTE*, E*] are last three cols, classifier was trained on TTE*,E*)
        tmp = os.path.join(args['Model_Eval_Path'], 'Classif_Outputs_and_Labels.hdf5')
        Save_to_hdf5(tmp, val_outputs, 'val_outputs')
        Save_to_hdf5(tmp, test_outputs, 'test_outputs')
        Save_to_hdf5(tmp, valid_df['E*'], 'valid_E*')
        Save_to_hdf5(tmp, valid_df['TTE*'], 'valid_TTE*')
        Save_to_hdf5(tmp, test_df['E*'], 'test_E*')
        Save_to_hdf5(tmp, test_df['TTE*'], 'test_TTE*')
        
        
        # %% 13. Run Cox models
        # 1. Fit a Cox model on the VALIDATION set, evaluate on TEST set
        # 2. convert risk prediction (0-1) to survival curves per subject
        # 3. measure concordance, AUPRC, AUROC, etc.
        # Note: Cox models train on un-horizoned labels; classifiers train on horizoned labels
        # (so this task uses all the time data, but the classifiers can only handle the one target)

        # build CoxPH curves on validation data
        zxcv = CoxPHSurvivalAnalysis() 
        a = valid_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
        b = valid_df['Mort_TTE'].to_numpy()
              
        tmp = np.array([ (a[k],b[k]) for k in range(a.shape[0]) ], dtype = [('event',bool),('time',float)] )
        zxcv.fit(np.expand_dims(val_outputs,-1), tmp   )
        
        # prep evaluation data - be sure to use actual time/event, not horizon
        if (args['Eval_Dataloader'] == 'Train'):
            disc_y_e = train_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
            disc_y_t = train_df['Mort_TTE'].to_numpy()
        elif (args['Eval_Dataloader'] == 'Validation'):
            disc_y_e = valid_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
            disc_y_t = valid_df['Mort_TTE'].to_numpy()
        else:
            disc_y_e = test_df['Mort_Event'].to_numpy().astype(bool) # Event as bool
            disc_y_t = test_df['Mort_TTE'].to_numpy()
    
        # %% 14. Match PyCox analyses
        # sample survival functions at a set of times (to compare to direct survival models)
        upper_time_lim = max( b ) # the direct survival models don't predict beyond the validation end time
        sample_time_points = np.linspace(0, upper_time_lim, 100).squeeze()
        
        surv_funcs = zxcv.predict_survival_function(np.expand_dims(test_outputs,-1))
        surv = np.squeeze(  np.array([k(sample_time_points) for k in surv_funcs]))
        
        disc_y_e = disc_y_e.astype(int).squeeze()
        disc_y_t = np.array([np.argmax(sample_time_points>=k) if k<=upper_time_lim else len(sample_time_points)-1 for k in disc_y_t]) # bin times. none should appear in bin 0
        
        surv_df = pd.DataFrame(np.transpose(surv)) 
        
        # %% 15. Save out everything we need to recreate evaluation off-cluster: 
        hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
        Save_to_hdf5(hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(hdf5_path, disc_y_e, 'disc_y_e') # what really happened
        Save_to_hdf5(hdf5_path, disc_y_t, 'disc_y_t') # when it really happened, discretized
        Save_to_hdf5(hdf5_path, test_df['E*'].to_numpy(), 'Test E*')
        Save_to_hdf5(hdf5_path, test_df['Age'], 'Age')
        Save_to_hdf5(hdf5_path, test_df['Is_Male'], 'Is_Male')
        Save_to_hdf5(hdf5_path, surv, 'surv')

        print('Model_Runner: Generated survival curves. Total time elapsed: ' + str(time.time()-start_time) )
        
        # %% 16. evlauations
        
        # Save out KM. Add bootstrapping (20x). Save KM values out separately
        Gen_KM_Bootstraps(surv, disc_y_e, disc_y_t, sample_time_points, args)

        # Concordance and Brier Score 
        time_points = [1,2,5,10,999]
        # across all ECG
        Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e, time_points, sample_time_points, args)
        
        # bootstrap: 1 ECG per patient x 20
        # Gen_Concordance_Brier_PID_Bootstrap(Data, args, disc_y_t, disc_y_e, surv_df, sample_time_points, time_points)
        
        # AUROC and AUPRC
        time_points = [1,2,5,10] # 999 doesn't work for AUROC
        Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points, args)
        print_classifier_ROC(test_correct_outputs, test_outputs)
        
        # histogram
        save_histogram(sample_time_points, disc_y_t, surv, args)
        
        print('Model_Runner: Finished evaluation. Total time elapsed: ' + str(time.time()-start_time) )
        
        
        # %% For downstream work
        # %% 20. Save out some more things 
        
        if ('Multimodal_Out' in args.keys()):
            if (args['Multimodal_Out'] == 'True'):

                
                if (args['Eval_Dataloader'] == 'Test'):
                    eval_key = 'y_test'
                # save the model's predictions at the 1 month mark
                
                
                multim_path = os.path.join(args['Model_Eval_Path'], 'Multimodal_Prediction_Out_'+eval_key+'.csv')

                # PID, TTE*, E* @ column inds [-3,-2,-1], respectively
                surv_target_index = np.argmin( abs(sample_time_points - 30/365)) # mark the 1 month surv time point

                # next line differs from _Survclass: Data['y_test'][:,-2], which is usually TTE*, gets discretized. 
                temp = np.vstack( (Data[eval_key][:,-3],1-surv[:,surv_target_index],disc_y_e,Data[eval_key][:,int(args['y_col_test_time'])], Data[eval_key][:,-1], Data[eval_key][:,5], Data[eval_key][:,1], Data[eval_key][:,2], test_outputs))
                temp = np.transpose(temp)
                headers = 'PID, surv_1month_output, E, TTE, E*, SID, Age, Is_Male, classif_outputs'
                np.savetxt(multim_path, temp, header=headers, delimiter = ',')
                
            
                if ('Neg_One_Out' in args.keys()):
                    # also save out the features
                    neg_one_path = os.path.join(args['Model_Eval_Path'], 'Model_Features_Out_'+eval_key+'.hdf5')
                    neg_one_out = my_model.Get_Features_Out(Which_Dataloader = eval_key)
                    Save_to_hdf5(neg_one_path, headers, 'MM_headers')
                    Save_to_hdf5(neg_one_path, temp, 'MM_Out')
                    Save_to_hdf5(neg_one_path, neg_one_out, 'neg_one_out')
                    
    
    #%% Test?
if __name__ == '__main__':
   main()