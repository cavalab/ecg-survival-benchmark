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
    
    # Y covariate indices get passed as args too
    val_covariate_col_list, test_covariate_col_list = get_covariates(args)
    
    if ('Eval_Dataloader' not in args.keys()): # This lets you evaluate the model on its validation set instead of test set
        args['Eval_Dataloader'] = 'Test'
    
    # %% Process data: Load, Clean, Split
    Data, Train_Col_Names, Test_Col_Names = Load_Data(args)       # Data is a dict, is passed by reference
    Clean_Data(Data, args)       # remove TTE<0 and NaN ECG
    Apply_Horizon(Data, args)    # classifiers need to compact TTE and E into a single value, E*. Augments Data['y_'] for model train/runs without overwriting loaded information.
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
        
            
    # %% 9. set up model folders if they  don't exist
    set_up_train_folders(args)

    # %% 10. Select model, (maybe) load an existing model. ask for training (runs eval after training reqs met)
    print('Model_Runner: Got to init. Total time elapsed: ' ,'{:.2f}'.format(time.time()-start_time) )
    
    my_model = GenericModelClassifierCox.GenericModelClassifierCox(args, Data)
    
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
        Save_to_hdf5(tmp, Data['y_valid'], 'y_valid')
        Save_to_hdf5(tmp, Data['y_test'], 'y_test')
        
        # %% 13. Run Cox models
        # 1. Fit a Cox model on the VALIDATION set, evaluate on TEST set
        # 2. convert risk prediction (0-1) to survival curves per subject
        # 3. measure concordance, AUPRC, AUROC, etc.
        # Note: Cox models train on un-horizoned labels; classifiers train on horizoned labels
        # (so this task uses all the time data, but the classifiers can only handle the one target)

        # build CoxPH curves on validation data
        zxcv = CoxPHSurvivalAnalysis() 
        a = Data['y_valid'][:,[int(args['y_col_train_event']) ]].astype(bool)   
        b = Data['y_valid'][:,[int(args['y_col_train_time']) ]]                 
        tmp = np.array([ (a[k],b[k][0]) for k in range(a.shape[0]) ], dtype = [('event',bool),('time',float)] )
        zxcv.fit(np.expand_dims(val_outputs,-1), tmp   )
        
        # prep evaluation data - be sure to use actual time/event, not horizon
        if (args['Eval_Dataloader'] == 'Train'):
            disc_y_e = Data['y_train'][:,[int(args['y_col_train_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_train'][:,[int(args['y_col_train_time']) ]] # prep time
        elif (args['Eval_Dataloader'] == 'Validation'):
            disc_y_e = Data['y_valid'][:,[int(args['y_col_train_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_valid'][:,[int(args['y_col_train_time']) ]] # prep time
        else:
            disc_y_e = Data['y_test'][:,[int(args['y_col_test_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_test'][:,[int(args['y_col_test_time']) ]] # prep time
    
        # %% 14. Match PyCox analyses
        # sample survival functions at a set of times (to compare to direct survival models)
        upper_time_lim = max( b )[0] # the direct survival models don't predict beyond the validation end time
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
        Save_to_hdf5(hdf5_path, Data['y_test'], 'y_test')
        Save_to_hdf5(hdf5_path, surv, 'surv')
        Save_to_hdf5(hdf5_path, Test_Col_Names + ['PID', 'TTE*', 'E*'], 'Test_Col_Names')

        print('Model_Runner: Generated survival curves. Total time elapsed: ' + str(time.time()-start_time) )
        
        # %% 16. evlauations
        
        # Save out KM. Add bootstrapping (20x). Save KM values out separately
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
        
        
        # %% For downstream work
        # %% Maybe save out 1-month performance and/or model features
        
        if ('Multimodal_Out' in args.keys()):
            if (args['Multimodal_Out'] == 'True'):
                multim_path = os.path.join(args['Model_Eval_Path'], 'Multimodal_Out.csv')
                
                # PID, TTE*, E* @ column inds [-3,-2,-1], respectively
                temp = np.vstack( (Data['y_test'][:,-3],test_outputs,test_correct_outputs,Data['y_test'][:,-2], Data['y_test'][:,-1] )  )
                temp = np.transpose(temp)
                headers = 'PID, test_outputs, test_correct_output, TTE, E'
                np.savetxt(multim_path, temp, header=headers, delimiter = ',')
                
                Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, [30/365, 1, 2, 5, 10], sample_time_points, args)
            
                if ('Neg_One_Out' in args.keys()):
                    neg_one_path = os.path.join(args['Model_Eval_Path'], 'Model_Features_Out.hdf5')
                    neg_one_out = my_model.Get_Features_Out(Which_Dataloader = args['Eval_Dataloader'])
                    
                    Save_to_hdf5(neg_one_path, headers, 'MM_headers')
                    Save_to_hdf5(neg_one_path, temp, 'MM_Out')
                    Save_to_hdf5(neg_one_path, neg_one_out, 'neg_one_out')
                    
        # # PID = Data_Y_Backup['y_test'][:,0]
        # # test_outputs # softmax
        # # test_correct_outputs # label passed
        # # TTE = Data_Y_Backup['y_test'][:,3]
        # # TTE = Data_Y_Backup['y_test'][:,4]
        
        # # %%21. Save out some more details
        
        # # 1. What is distribution of ECG count per PID in last T time?
        # x = 0.05*np.arange(21)
        # save_out = [x]
        
        # for time_lim in [1/52,1/12,1,999]:
        #     subset = Data_Y_Backup['y_train'][np.where(Data_Y_Backup['y_train'][:,3] <= time_lim)[0],:]
        #     unique_PID = np.unique(subset[:,0])
        #     counts = [np.where(subset[:,0] == k)[0].shape[0] for k in unique_PID]
        #     event = [Data_Y_Backup['y_train'][np.where(subset == k)[0][0],4] for k in unique_PID  ]
        #     event = np.array(event).astype(int)
        #     counts = np.array(counts).astype(int)
            
        #     y1 = np.quantile(counts[np.where(event==1)],x)
        #     y2 = np.quantile(counts[np.where(event==0)],x)
        #     save_out.append(y1)
        #     save_out.append(y2)
            
        #     plt.figure()
        #     plt.scatter(x*100,y2)
        #     plt.scatter(x*100,y1)
        #     plt.legend(['event=0 med '+str(np.median(y2))+' u '+'{0:.2f}'.format(np.mean(y2)),'event=1 med '+str(np.median(y1)) + ' u '+'{0:.2f}'.format(np.mean(y1))])
        #     plt.xlabel('percentile')
        #     plt.ylabel('count')
        #     plt.title('count per PID | TTE < '+str(time_lim))
            
        # characteristics_path = os.path.join(temp_path, 'Train_Characteristics.csv')
        # headers = 'percentile, 1-wk event 1, 1-wk event 0, 1-mo event 1, 1-mo event 0,1-yr event 1, 1-yr event 0, all-t event 1, all-t event 0'
        # temp = np.array(save_out).transpose()
        # np.savetxt(characteristics_path, temp, header=headers, delimiter = ',')
        
    #%% Test?
if __name__ == '__main__':
   main()