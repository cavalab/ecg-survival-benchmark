# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:23:50 2024

@author: CH242985
"""

import os
import h5py
import shutil
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
from MODELS.Support_Functions import Save_to_hdf5
import time

# evaluation wrappers
from Model_Runner_Support import Gen_KM_Bootstraps
from Model_Runner_Support import Gen_Concordance_Brier_No_Bootstrap
from Model_Runner_Support import Gen_Concordance_Brier_PID_Bootstrap
from Model_Runner_Support import Gen_AUROC_AUPRC
from Model_Runner_Support import print_classifier_ROC
from Model_Runner_Support import save_histogram

# %%
Train_Folder = 'MIMICIV_Multimodal_Subset'
Model_Name = 'InceptionClass_082424_10_2348968'
Eval_Folder = 'MIMICIV_Multimodal_Subset Test_Folder'

args = {}
args['y_col_train_time'] = '3'
args['y_col_train_event'] = '4'
args['y_col_test_time'] = '3'
args['y_col_test_event'] = '4'
args['Eval_Dataloader'] = 'Test'

# Covariates:
# MIMICIV: Age 1, Gender 2  
# Code15: Age 2, Gender 5
# MIMIC All:  [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

args['name_aug'] = '_DEM_COX' # how should we call this model variant?    
args['val_covariate_col_list'] = '[1,2]' 
args['test_covariate_col_list'] = '[1,2]'

# args['name_aug'] = '_FullMM_COX' # how should we call this model variant?    
# args['val_covariate_col_list'] = '[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]' 
# args['test_covariate_col_list'] = '[1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]'


# %% 10.5 Decide on which rows to use
val_covariate_col_list = []
if ('val_covariate_col_list' in args.keys()):
    val_covariate_col_list = [int(k) for k in args['val_covariate_col_list'][1:-1].split(',')]
    
test_covariate_col_list = []
if ('test_covariate_col_list' in args.keys()):
    test_covariate_col_list = [int(k) for k in args['test_covariate_col_list'][1:-1].split(',')]


# %% go

start_time = time.time()

if os.path.isfile(os.path.join(os.getcwd(), 'Trained_Models', Train_Folder, Model_Name, 'EVAL', Eval_Folder, 'Classif_Outputs_and_Labels.hdf5')):
    
    
    # the classifier has the right output for us to proceed
    src = os.path.join(os.getcwd(), 'Trained_Models', Train_Folder, Model_Name)
    dst = os.path.join(os.getcwd(), 'Trained_Models', Train_Folder, Model_Name+args['name_aug'])
    
    if (os.path.isdir(dst)):
        print('Path already exists. no copying.')
    else:
        shutil.copytree(src,dst)
        
    with h5py.File(os.path.join(dst, 'Eval',Eval_Folder, 'Classif_Outputs_and_Labels.hdf5'), "r") as f:
        Data = {}
        test_outputs = f['test_outputs'][()]
        val_outputs = f['val_outputs'][()]
        Data['y_test'] = f['y_test'][()]
        Data['y_valid'] = f['y_valid'][()]
    
    args['Model_Eval_Path'] = os.path.join(dst, 'Eval', Eval_Folder)
        
        
    # okay, now we can repeat the cox measurements
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
    
    
    # Augment model outputs on validation set with new covariates
    if (len(val_outputs.shape)==1):
        val_outputs = np.expand_dims(val_outputs,-1)
    dem_appended_val_outputs = np.hstack( (val_outputs, Data['y_valid'][:,val_covariate_col_list]))
    
    zxcv.fit(dem_appended_val_outputs, tmp   ) # n x 1
    
    # prep evaluation data - evaluate on if/when things actually happen
    disc_y_e = Data['y_test'][:,[int(args['y_col_test_event']) ]].astype(bool) # prep event
    disc_y_t = Data['y_test'][:,[int(args['y_col_test_time']) ]] # prep time

    # %% 14. Prep everything to match PyCox analysis
    # sample survival functions at a set of times (to compare to direct survival moels)
    upper_time_lim = max( b )[0] # the fit is limited to validation end times, so do 100 bins of tha
    sample_time_points = np.linspace(0, upper_time_lim, 100).squeeze()

    # Augment model outputs on test set with new covariates
    if (len(test_outputs.shape)==1):
        test_outputs = np.expand_dims(test_outputs,-1)
    dem_appended_test_outputs = np.hstack( (test_outputs, Data['y_test'][:,test_covariate_col_list]))
    
    surv_funcs = zxcv.predict_survival_function(dem_appended_test_outputs)
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
    # Save_to_hdf5(hdf5_path, Test_Col_Names + ['PID', 'TTE*', 'E*'], 'Test_Col_Names') # already there

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
    # print_classifier_ROC(test_correct_outputs, test_outputs)
    
    # histogram
    save_histogram(sample_time_points, disc_y_t, surv, args)
    
    print('Model_Runner: Finished evaluation. Total time elapsed: ' + str(time.time()-start_time) )
    