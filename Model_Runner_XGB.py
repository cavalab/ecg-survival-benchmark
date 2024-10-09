# 1. Pull data, split, continue as usual until train


# %% Imports; Support functions before main functions...

# handle pycox folder requirement FIRST
import os 
os.environ['PYCOX_DATA_DIR'] = os.path.join(os.getcwd(),'Mandatory_PyCox_Dir') # next line requires this. it's a bad init call.



from Model_Runner_Support import get_covariates

from Model_Runner_Support import Load_Data
from Model_Runner_Support import Clean_Data
from Model_Runner_Support import Apply_Horizon
from Model_Runner_Support import Split_Data
from Model_Runner_Support import DebugSubset_Data
from Model_Runner_Support import set_up_train_folders
from Model_Runner_Support import set_up_test_folders

# evaluation wrappers
from Model_Runner_Support import Gen_KM_Bootstraps
from Model_Runner_Support import Gen_Concordance_Brier_No_Bootstrap
from Model_Runner_Support import Gen_Concordance_Brier_PID_Bootstrap
from Model_Runner_Support import Gen_AUROC_AUPRC
from Model_Runner_Support import print_classifier_ROC
from Model_Runner_Support import save_histogram

from MODELS import GenericModelSurvClass


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

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# scipy compatability...
import scipy
scipy.integrate.simps = scipy.integrate.simpson

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
    
    # 
def Run_Model(args):
    # input is paired dict of strings named args
    start_time = time.time()
    
    # %% 1. CUDA check
    # CUDA
    # for i in range(torch.cuda.device_count()):
    #    print(torch.cuda.get_device_properties(i).name)
       
    # if (torch.cuda.is_available() == False):
    #     print('No CUDA. Exiting.')
    #     exit()
       
    # %% 2. Arg Processing
    # Grab model name. No point in proceeding without it.
    if ('Model_Name' not in args.keys()):
        print('Model_Name not specified - cant train or pull models')
        exit()

    Model_Type = args['Model_Name'].split('_')[0]
    args['Model_Type'] = Model_Type
    
    if ('Train_Folder' not in args.keys()):
        print('Train_Folder not specified - cant train or pull models')
        exit()
    
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
    
    # Y covariate indices get passed here
    val_covariate_col_list, test_covariate_col_list = get_covariates(args)
    
    # %% Process data: Load, Clean, Split
    Data, Train_Col_Names, Test_Col_Names = Load_Data(args)       # Data is a dict, so passed by reference from now on
    Clean_Data(Data, args)       # remove TTE<0 and NaN ECG
    Apply_Horizon(Data, args)    # classifiers need to compact TTE and E into a single value, E*. Augments Data['y_'] for model train/runs without overwriting loaded information.
    Split_Data(Data)             # splits 'train' data 80/20 into train/val by PID
    DebugSubset_Data(Data, args) # If args['debug'] == True, limits Data[...] to 1k samples each of tr/val/test.
    
    # dump x
    for key in ['train', 'valid', 'test']:
        x_key = 'x_'+key
        Data[x_key] = []
    
    # augment data with reshaped covariates
    for key in ['train', 'valid']:
        y_key = 'y_'+key
        z_key = 'z_'+key # for covariates
        Data[z_key] = Data[y_key][:,val_covariate_col_list]
        
    for key in ['test']:
        y_key = 'y_'+key
        z_key = 'z_'+key # for covariates
        Data[z_key] = Data[y_key][:,test_covariate_col_list]
        
            
    # %% 9. set up trained model folders if they  don't exist
    set_up_train_folders(args)

    # %% 10. Select model, (maybe) load an existing model. ask for training (runs eval after training reqs met)
    
    
    # This is where things change
    
    # from https://xgboost.readthedocs.io/en/stable/get_started.html
    asdf = XGBClassifier(n_estimators=100, learning_rate=0.1, objective='binary:logistic')
    
    
    # do we need to normalize data first?
    # from https://xgboosting.com/xgboost-min-max-scaling-numerical-input-features/
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Data['z_train'] = scaler.fit_transform(Data['z_train'])
    Data['z_valid'] = scaler.transform(Data['z_valid'])
    Data['z_test'] = scaler.transform(Data['z_test'])
    
    asdf.fit(Data['z_train'], Data['y_train'][:,-1])
    
    
    
    # print('Model_Runner: Got to init. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    
    # asdf = GenericModelSurvClass.GenericModelSurvClass(args, Data)
    
    # print('Model_Runner:  Got to Train. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    # if( ('Load' in args.keys())):
    #     asdf.Load(args['Load'])
    # asdf.train()
        

    if ('Test_Folder' in args.keys()):
    # %% 11. Generate and save out results   
    # Per new plan 8/29/24, classifier models save out the softmax'd outputs for class '1' (event happened) on VAL and TEST
    # Those will be stored along with data[y_] for VAL and TEST
    # ... so we can add covariates to the cox regressions (built on VAL) without engaging GPU
        print('got to eval. Total Time elapsed: ' ,'{:.2f}'.format(time.time()-start_time))
        
        # get model outputs for test, validation sets (unshuffled)
        if ('Eval_Dataloader' not in args.keys()): # This lets you evaluate the model on its validation set instead of test set
            args['Eval_Dataloader'] = 'Test'
            
        if args['Eval_Dataloader'] == 'Test':
            test_outputs =  asdf.predict_proba(Data['z_test'])[:,1]
            test_correct_outputs = Data['y_test'][:,-1] # E*, horizoned event, is at -1
        if args['Eval_Dataloader'] == 'Train':
            test_outputs =  asdf.predict_proba(Data['z_train'])[:,1]
            test_correct_outputs = Data['y_train'][:,-1] # E*, horizoned event, is at -1
        if args['Eval_Dataloader'] == 'Valid':
            test_outputs =  asdf.predict_proba(Data['z_valid'])[:,1]
            test_correct_outputs = Data['y_valid'][:,-1] # E*, horizoned event, is at -1
            
        # get validation outputs anyway for Cox model fit
        val_outputs =  asdf.predict_proba(Data['z_valid'])[:,1]
        val_correct_outputs = Data['y_valid'][:,-1] # E*, horizoned event, is at -1
            

        # adjust output formats
        val_outputs  = np.squeeze(val_outputs)
        test_outputs = np.squeeze(test_outputs)
        
        # softmax the outputs (no need in XGB)
        # predict_proba is already 0-1, so don't need to softmax
        # val_outputs = np.array([softmax(k)[1] for k in val_outputs])
        # test_outputs = np.array([softmax(k)[1] for k in test_outputs])
        
        
        # --- From here, everything should go as before
        
        # Set up Folders
        set_up_test_folders(args)
        
        
        # Save out smx val/test model outputs + the labels ( [PID, TTE*, E*] are last three cols, model was trained on TTE*,E*)
        # From this we can recreate Cox models later
        tmp = os.path.join(args['Model_Eval_Path'], 'Classif_Outputs_and_Labels.hdf5')
        Save_to_hdf5(tmp, val_outputs, 'val_outputs')
        Save_to_hdf5(tmp, test_outputs, 'test_outputs')
        Save_to_hdf5(tmp, Data['y_valid'], 'y_valid')
        Save_to_hdf5(tmp, Data['y_test'], 'y_test')
        
        # %% 13. Run Cox models
        # fit a Cox model on the VALIDATION set, evaluate on TEST set
        # 1. convert risk prediction (0-1)  to survival curves per subject
        # 2. measure concordance, brier, AUPRC, AUROC, etc.
        # NoteL Cox models are built on un-horizoned labels, even if the classifiers are trained on horizoned labels
        # (so this task uses all the time data, but the classifiers can only handle the one target)

        # build CoxPH curves on validation data
        zxcv = CoxPHSurvivalAnalysis() 
        a = Data['y_valid'][:,[int(args['y_col_train_event']) ]].astype(bool)   # CoxPH is built on if/when the event actually happened
        b = Data['y_valid'][:,[int(args['y_col_train_time']) ]]                 # shouldn't this also be limited?
        tmp = np.array([ (a[k],b[k][0]) for k in range(a.shape[0]) ], dtype = [('event',bool),('time',float)] )
        zxcv.fit(np.expand_dims(val_outputs,-1), tmp   )
        
        # prep evaluation data - evaluate on if/when things actually happen
        if (args['Eval_Dataloader'] == 'Train'):
            disc_y_e = Data['y_train'][:,[int(args['y_col_train_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_train'][:,[int(args['y_col_train_time']) ]] # prep time
            
        elif (args['Eval_Dataloader'] == 'Validation'):
            disc_y_e = Data['y_valid'][:,[int(args['y_col_train_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_valid'][:,[int(args['y_col_train_time']) ]] # prep time
            
        else:
            disc_y_e = Data['y_test'][:,[int(args['y_col_test_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_test'][:,[int(args['y_col_test_time']) ]] # prep time
    
        # %% 14. Prep everything to match PyCox analysis
        # sample survival functions at a set of times (to compare to direct survival moels)
        upper_time_lim = max( b )[0] # the fit is limited to validation end times, so do 100 bins of tha
        sample_time_points = np.linspace(0, upper_time_lim, 100).squeeze()
        
        surv_funcs = zxcv.predict_survival_function(np.expand_dims(test_outputs,-1))
        surv = np.squeeze(  np.array([k(sample_time_points) for k in surv_funcs]))
        
        disc_y_e = disc_y_e.astype(int).squeeze()
        disc_y_t = np.array([np.argmax(sample_time_points>=k) if k<=upper_time_lim else len(sample_time_points)-1 for k in disc_y_t]) # bin times. none should appear in bin 0
        
        surv_df = pd.DataFrame(np.transpose(surv)) 
        
        # %% 15. Save out everything we need to recreate evaluation: 
        hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
        Save_to_hdf5(hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(hdf5_path, disc_y_e, 'disc_y_e') # what really happened
        Save_to_hdf5(hdf5_path, disc_y_t, 'disc_y_t') # when it really happened, discretized
        Save_to_hdf5(hdf5_path, Data['y_test'], 'y_test')
        Save_to_hdf5(hdf5_path, surv, 'surv')
        Save_to_hdf5(hdf5_path, Test_Col_Names + ['PID', 'TTE*', 'E*'], 'Test_Col_Names')

        print('Model_Runner: Generated survival curves. Total time elapsed: ' + str(time.time()-start_time) )
        
        # %% 16. evlauations
        
        # Save out KM. Add bootstrapping (20x). Saves KM values out separately in case you want to recreate that.
        Gen_KM_Bootstraps(surv, disc_y_e, disc_y_t, sample_time_points, args)

        # Concordance and Brier Score 
        time_points = [1,2,5,10,999]
        # across all ECG
        Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e, time_points, sample_time_points, args)
        # bootstrap: 1 ECG per patient x 20
        Gen_Concordance_Brier_PID_Bootstrap(Data, args, disc_y_t, disc_y_e, surv_df, sample_time_points, time_points)
        
        # AUROC and AUPRC
        time_points = [1,2,5,10] # 999 doesn't work for AUROC
        Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points, args)
        print_classifier_ROC(test_correct_outputs, test_outputs)
        
        # histogram
        save_histogram(sample_time_points, disc_y_t, surv, args)
        
        print('Model_Runner: Finished evaluation. Total time elapsed: ' + str(time.time()-start_time) )
        
        
    #%% Test?
if __name__ == '__main__':
   main()