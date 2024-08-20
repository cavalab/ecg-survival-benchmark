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
from pycox.utils import kaplan_meier

from MODELS.Support_Functions import get_surv_briercordance 
from MODELS.Support_Functions import get_AUROC_AUPRC
from MODELS.Support_Functions import Save_to_hdf5

from MODELS import Support_Functions
from MODELS import InceptionTimeRegression_PyCox
from MODELS import Ribeiro_Regression_PyCox
from MODELS import LSTMReg_PyCox
from MODELS import TimesNetReg_PyCox

import h5py
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
    
    #%%0. Toggleable speedup for debug
    debug = False
    if ('debug' in args.keys()):
        if args['debug'] == 'True':
            debug = True
            sub_len = 1000
            if (debug):
                print("Debug! only using "+str(sub_len)+' elems of tr/val/test!')

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
    
    # %% 2. Data load - NeurIPS simplified. Assume we are pulling from 'train_folder' and 'test_folder'
    Data = {}
    datapath1 = os.path.dirname(os.getcwd()) # cleverly jump one one folder without referencing \\ (windows) or '/' (E3)
    with h5py.File(os.path.join(datapath1,'HDF5_DATA',args['Train_Folder'],'TRAIN_DATA','Train_Data.hdf5'), "r") as f:
        Data['x_train'] = f['x'][()]
        Data['y_train'] = f['y'][()]
    
    with h5py.File(os.path.join(datapath1,'HDF5_DATA',args['Test_Folder'],'TEST_DATA','Test_Data.hdf5'), "r") as f:
        Data['x_test'] = f['x'][()]
        Data['y_test'] = f['y'][()] 
        
    print('Loaded Train and Test. Total time elapsed: ' + str(time.time()-start_time) )
    
    # %% 3.1 Data clean
    # Sometimes time-to-event is '-1.0' meaning no follow up time.
    # Sometimes TTE < 0 (recording error?)
    # Find and trash those indices    
    for key in ['y_train', 'y_test']:
        if ( (key in Data.keys()) and (len(Data[key].shape) > 1) ):
            x_key = 'x' + key[1:]
            
            # mark negative TTE
            neg_inds = np.where(Data[key][:,int(args['y_col_train_time'])] < 0)[0]
            inds_to_del = neg_inds.tolist()
            
            # mark nan traces (5x faster)
            for i in range(Data[x_key].shape[0]):
                if (np.isnan(Data[x_key][i]).any()):
                    if i not in inds_to_del:
                        inds_to_del.append(i)
                        
            # mark nan traces (pre 8/14/24)
            # nan_inds = np.where(np.sum(np.sum(np.isnan(Data[x_key]),axis=2),axis=1) > 0)[0]
            # for k in nan_inds:
            #     if k not in inds_to_del:
            #         inds_to_del.append(k)
                     
            # remove data, avoid calling np.delete on Data cause that doubles RAM - just select the indices to keep instead
            if (len(inds_to_del) > 0):
                print('removing ' + str(len(inds_to_del)) + ' inds with nan or negative time')
                inds_to_keep = np.delete( np.arange(Data[key].shape[0]), inds_to_del )
                Data[x_key] = Data[x_key][inds_to_keep] 
                Data[key] = Data[key][inds_to_keep] 
                
    print('Checked data for negative time and nan ECG. Total time elapsed: ' + str(time.time()-start_time) )                
            
    # %% 3.2 Backupdata labels for evaluation 
    Data_Y_Backup = {}
    if ( ('y_train' in Data.keys()) and (len(Data['y_train'].shape) > 1) ):
        Data_Y_Backup['y_train'] = np.copy(Data['y_train'])
    
    if ( ('y_test' in Data.keys()) and (len(Data['y_test'].shape) > 1) ):
        Data_Y_Backup['y_test'] = np.copy(Data['y_test'])
        
    # %% 3.3 Prep the data you'll send over to the model
    if ( ('y_train' in Data.keys()) and (len(Data['y_train'].shape) > 1) ):
        Data['y_train'] = Data['y_train'][:,[int(args['y_col_train_time']),int(args['y_col_train_event'])]]

    if ( ('y_test' in Data.keys()) and (len(Data['y_test'].shape) > 1) ):
        Data['y_test'] = Data['y_test'][:,[int(args['y_col_test_time']),int(args['y_col_test_event'])]]

    # %% 4. Split loaded TRAIN data into Tr/Val/(Test = 0) by PID
    if ('y_train' in Data.keys()):
        
        # NeurIPS version - split Training dataset 80 / 20 into Tr/Val
        TR = 80
        VA = 20
        TE = 00

        # Per ID, find matching data rows      
        Subj_IDs = Data_Y_Backup['y_train'][:,0]            
        Subj_IDs_Unique = np.unique(Subj_IDs)
        
        #Speedup 08/14/24
        Subj_ID_to_Rows_Dict = {} # map ID to rows
        for ind,val in enumerate(Subj_IDs):
            if val in Subj_ID_to_Rows_Dict.keys():
                Subj_ID_to_Rows_Dict[val].append(ind)
            else:
                Subj_ID_to_Rows_Dict[val] = [ind]
                
        # Subj_ID_to_Rows_Dict = {} # map ID to rows
        # for k in Subj_IDs_Unique:
        #     Subj_ID_to_Rows_Dict[k] = np.where(Subj_IDs == k)[0] 

        # Split
        Train_Inds, Val_Inds, Test_Inds = Support_Functions.Data_Split_Rand( [k for k in range(len(Subj_IDs_Unique))], TR, VA, TE)

        # Okay, now that we've split unique patient IDs we need to convert that back to ECG rows
        # Train_Inds_ECG = []
        # for k in Train_Inds:
        #     Train_Inds_ECG = Train_Inds_ECG + [ m for m in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[k]] ]
        # Faster equivalent. 0.05s instead of 213.
        Train_Inds_ECG  = [Row for PID in Train_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        
        # Val_Inds_ECG = []
        # for k in Val_Inds:
        #     Val_Inds_ECG = Val_Inds_ECG + [ m for m in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[k]] ]
        # Faster Equivalent
        Val_Inds_ECG  = [Row for PID in Val_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        
        # Test_Inds_ECG = []
        # for k in Test_Inds:
        #     Test_Inds_ECG = Test_Inds_ECG + [ m for m in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[k]] ]
        # Faster Equivalent
        Test_Inds_ECG = [Row for PID in Test_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        
        Train_Inds = Train_Inds_ECG
        Val_Inds = Val_Inds_ECG
        Test_Inds = Test_Inds_ECG
            
        Data['x_valid'] = Data['x_train'][Val_Inds]
        Data['x_train'] = Data['x_train'][Train_Inds]
    
        Data['y_valid'] = Data['y_train'][Val_Inds]
        Data['y_train'] = Data['y_train'][Train_Inds]
        
        Data_Y_Backup['y_valid'] = Data_Y_Backup['y_train'][Val_Inds]
        Data_Y_Backup['y_train'] = Data_Y_Backup['y_train'][Train_Inds]
          
    else:
        print('No Data Split')
        
    print('Split Train into Train/Valid. Total time elapsed: ' + str(time.time()-start_time) )                
            
    # %% 5. set up trained model folders if they  don't exist

    # trained models live in /trained_models/[train_data_folder]/Resnet18_Bob/asdf.pt
    # evals live in /trained_models/[train_data_folder]/Resnet18_Bob/Eval/[Eval_data_folder]/output.csv    
    
    # Set up Trained_Models folder
    temp_path = os.path.join(os.getcwd(),'Trained_Models')
    if ( (os.path.exists(temp_path) == False)):
        os.mkdir(temp_path)
    
    # Set up Trained_Models/[Train_Folder]
    temp_path = os.path.join(temp_path, args['Train_Folder'])
    if ( (os.path.exists(temp_path) == False)):
        os.mkdir(temp_path)
        
    # Set up and remember Trained_Models/[Train Folder]/[Model_Name]
    Model_Folder_Path = os.path.join(temp_path, args['Model_Name'])
    if (os.path.exists(Model_Folder_Path) == False):
        os.mkdir(Model_Folder_Path)
    
    args['Model_Folder_Path'] = Model_Folder_Path # pass model folder path to model to save out checkpoints / eval
    
    
    # %% 6. Select model, (maybe) train, and run
    # # debug - cut down data to first 1k samples for faster runs
    if (debug):
        print('!!!debug!!! Reducing data size!')
        tr_inds = np.random.randint(0, Data['x_train'].shape[0], (sub_len))
        va_inds = np.random.randint(0, Data['x_valid'].shape[0], (sub_len))
        te_inds = np.random.randint(0, Data['x_test'].shape[0], (sub_len))
        
        Data['x_train'] = Data['x_train'][tr_inds,:]
        Data['x_valid'] = Data['x_valid'][va_inds,:]
        Data['x_test'] = Data['x_test'][te_inds,:]
        
        Data['y_train'] = Data['y_train'][tr_inds,:]
        Data['y_valid'] = Data['y_valid'][va_inds,:]
        Data['y_test'] = Data['y_test'][te_inds,:]
        
        Data_Y_Backup['y_train'] = Data_Y_Backup['y_train'][tr_inds,:] 
        Data_Y_Backup['y_valid'] = Data_Y_Backup['y_valid'][va_inds,:] 
        Data_Y_Backup['y_test'] = Data_Y_Backup['y_test'][te_inds,:] 
        
        
    print('Got to model init. Total time elapsed: ' + str(time.time()-start_time) )
    if (Model_Type == 'InceptionTimeReg'):
        asdf = InceptionTimeRegression_PyCox.InceptionTimeRegression_PyCox(args, Data)
    if (Model_Type == 'RibeiroReg'):
        asdf = Ribeiro_Regression_PyCox.Ribeiro_Regression_PyCox(args, Data)
    if (Model_Type == 'LSTMReg'):
        asdf = LSTMReg_PyCox.LSTMReg_PyCox(args, Data)
    if (Model_Type == 'TimesNetReg'):
        asdf = TimesNetReg_PyCox.TimesNetReg_PyCox(args, Data)
      
        
    print('Got to Train. Total time elapsed: ' + str(time.time()-start_time) )
    if( ('Load' in args.keys())):
        asdf.Load(args['Load'])
    asdf.Train()
        
    
    # %% Model Evaluation
    # To evaluate model we need: dicrete time points, discretized time to event, event 0/1, S(t) per time point
    
    if ('Test_Folder' in args.keys()):
        print('got to model eval. Total Time elapsed: ' + str(time.time()-start_time))
        
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
        
        
        # %% Generate and save out results
        
        # 1. Set up folders.
        # evals live in /trained_models/[train_data_folder]/Resnet18_Bob/Eval/[eval_data_folder]/output.csv
        
        # Set up .../Eval
        temp_path = os.path.join(Model_Folder_Path, 'EVAL')
        if (os.path.isdir(temp_path) == False):
            os.mkdir(temp_path)
            
        # Set up .../Eval/eval_data_folder
        temp_path = os.path.join(temp_path, args['Test_Folder'] + ' Test_Folder')
        if (os.path.isdir(temp_path) == False):
            os.mkdir(temp_path)

        # 2. Save out evaluation args (might differ from training args)
        import json
        path = os.path.join(temp_path, 'Eval_Args.txt')
        with open(path, 'w') as file:
             file.write(json.dumps(args)) # use `json.loads` to do the reverse
             
        # 3. Save out sample_time_points, testy_t, testy_e, s(t)
        outputs_hdf5_path = os.path.join(temp_path, 'Stored_Model_Output.hdf5')
        Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(outputs_hdf5_path, disc_y_t, 'disc_y_t')
        Save_to_hdf5(outputs_hdf5_path, disc_y_e, 'disc_y_e')
        Save_to_hdf5(outputs_hdf5_path, surv, 'surv')

        # %%  Generate survival curves for model and kaplan-meier, bootstrapping patient selection(100x)
        print('Got to KM bootstraps. Total time elapsed: ' + str(time.time()-start_time) )
        
        # Bootstrap
        km_outs = []
        mdl_outs = []
        for k in range(100):
            sample_inds = np.random.randint(0,surv.shape[0],(surv.shape[0]))
            km_out = kaplan_meier(events=disc_y_e[sample_inds], durations=disc_y_t[sample_inds]).to_numpy().tolist()
            mdl_out = np.mean(surv[sample_inds],axis=0)
            
            # sometimes don't get a last index on the KM - extend last value out so array sizes line up
            while (len(km_out) < surv.shape[1]):
                km_out.append(km_out[-1])
                
            km_outs.append(km_out)
            mdl_outs.append(mdl_out)
            
        mdl_outs = np.vstack(mdl_outs)
        km_outs = np.vstack(km_outs)
        
        # Now we want to plot and save out the median and 2.5-97.5th percentile per time point (100)
        mdl_int_low = [] # 
        mdl_int_high = []
        mdl_median = []
        km_int_low = []
        km_int_high = []
        km_median = []
        for k in range(km_outs.shape[1]):
            km_median.append(np.median(km_outs[:,k]))
            km_int_high.append(np.percentile(km_outs[:,k],97.5))
            km_int_low.append(np.percentile(km_outs[:,k],2.5))
            mdl_median.append(np.median(mdl_outs[:,k]))
            mdl_int_high.append(np.percentile(mdl_outs[:,k],97.5))
            mdl_int_low.append(np.percentile(mdl_outs[:,k],2.5))
            
        fig1, ax = plt.subplots()
        plt.plot(cuts, km_median)
        ax.fill_between(cuts, km_int_low, km_int_high, color='b', alpha=.1)
        plt.plot(cuts,mdl_median, color='r')
        ax.fill_between(cuts, mdl_int_low, mdl_int_high, color='r', alpha=.1)
        plt.legend(('KM','KM 2.5-97.5%','Model','Model 2.5-97.5%'))
        plt.xlabel('Years')
        plt.ylabel('Survival')
        plot_file_path = os.path.join(temp_path, 'KM vs Model 100xBS Survival Curve.pdf')
        fig1.savefig(plot_file_path)

        outputs_hdf5_path = os.path.join(temp_path, 'Surv_Outputs.hdf5')
        Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(outputs_hdf5_path, mdl_median, 'SF_mdl_median')
        Save_to_hdf5(outputs_hdf5_path, mdl_int_low, 'SF_int_low')
        Save_to_hdf5(outputs_hdf5_path, mdl_int_high, 'SF_mdl_int_high')
        Save_to_hdf5(outputs_hdf5_path, km_median, 'SF_km_median')
        Save_to_hdf5(outputs_hdf5_path, km_int_low, 'SF_km_int_low')
        Save_to_hdf5(outputs_hdf5_path, km_int_high, 'SF_km_int_high')
        
        print('Finished KM bootstraps. Starting all-ECG Brier and Concordance Total time elapsed: ' + str(time.time()-start_time) )
        
        # %% 6. Concordance and IPCW Brier Score across all ECGs
        concordance_store_all_ecg, ipcw_brier_store_all_ecg, chance_at_censored_point_all_ecg  = get_surv_briercordance(disc_y_t, disc_y_e, surv_df, [1,2,5,10,999], cuts)

        Save_to_hdf5(outputs_hdf5_path, concordance_store_all_ecg, 'concordance_store_all_ecg')
        Save_to_hdf5(outputs_hdf5_path, ipcw_brier_store_all_ecg, 'ipcw_brier_store_all_ecg')
        Save_to_hdf5(outputs_hdf5_path, chance_at_censored_point_all_ecg, 'chance_at_censored_point_all_ecg')
        
        fig1, ax = plt.subplots()
        plt.plot([0,max(cuts)],[.5,.5], '--') # concordance if guessing
        plt.plot(cuts, ipcw_brier_store_all_ecg)
        plt.plot(cuts, concordance_store_all_ecg)
        plt.ylim([0,1])
        plt.legend(['Concordance Chance','Brier = MSE','Concordance'])
        plt.xlabel('follow-up time (years)')
        plt.ylabel('Measure for time bin')
        plot_file_path = os.path.join(temp_path, 'briercordance, briercordance.pdf')
        fig1.savefig(plot_file_path)


    # %% now add brier and concordance, bootstrapping 20x and only keeping one ECG per subject.
        print('Got to bootstrap concordances. Total time elapsed: ' + str(time.time()-start_time) )
        
        # 1. find which data corresponds ot which patient ID
        if (args['Eval_Dataloader'] == 'Validation'):
            Subj_IDs = Data_Y_Backup['y_valid'][:,0]    
        else:
            Subj_IDs = Data_Y_Backup['y_test'][:,0]    
            
        Subj_IDs_Unique = np.unique(Subj_IDs)
        
        Subj_ID_to_Rows_Dict = {} # map ID to rows
        for k in Subj_IDs_Unique:
            Subj_ID_to_Rows_Dict[k] = np.where(Subj_IDs == k)[0] 
        
        bootstrap_briers = [] # list of lists
        bootstrap_concordances = [] # list of lists
        
        bootstraps = 20 
        Inds = [Subj_ID_to_Rows_Dict[k][0] for k in Subj_IDs_Unique]
        for b in range (bootstraps): # per bootstrap
            for i,s in enumerate(Subj_IDs_Unique): # parse all unique subject IDs
                tmp = Subj_ID_to_Rows_Dict[s] # get the evaluation output (time, event, survival odds) rows associated with subject s
                if (len(tmp) != 1):
                    Inds[i] = tmp[np.random.randint(0,len(tmp))] # if there are multiple options, pick a random one
    
            # measure concrodance, brier, and record 'chance' (% of cases recording a mortality by time T, ignoring censoring)
            concordance_score, ipcw_brier_score, chance_at_censored_point  = get_surv_briercordance(disc_y_t[Inds], disc_y_e[Inds], surv_df.iloc[:,Inds], [1,2,5,10,999], cuts)

            bootstrap_briers.append(ipcw_brier_score)
            bootstrap_concordances.append(concordance_score)
            
        Save_to_hdf5(outputs_hdf5_path, bootstrap_briers, 'bootstrap_briers')
        Save_to_hdf5(outputs_hdf5_path, bootstrap_concordances, 'bootstrap_concordances')
        print('Finished bootstrap concordance. Heading to AUC. Total time elapsed: ' + str(time.time()-start_time) )
            
        # %% AUROC and AUPRC and chance
        AUROC_store, AUPRC_store, Chance_At_Age = get_AUROC_AUPRC(disc_y_t, disc_y_e, surv, [1,2,5,10], cuts)
        
        fig1, ax = plt.subplots()
        plt.plot([0,max(cuts)],[.5,.5], '--')
        plt.plot(cuts,Chance_At_Age,'--')
        plt.plot(cuts, AUROC_store)
        plt.plot(cuts, AUPRC_store)
        plt.ylim([0,1])
        plt.legend(['AUROC Chance' , 'Chance at time','AUROC','AUPRC'])
        plt.xlabel('follow-up time (years)')
        plt.ylabel('Measure for time bin')
        plot_file_path = os.path.join(temp_path, 'AUROC, AUPRC.pdf')
        fig1.savefig(plot_file_path)

        Save_to_hdf5(outputs_hdf5_path, AUROC_store, 'AUROC')
        Save_to_hdf5(outputs_hdf5_path, AUPRC_store, 'AUPRC')

        # %% save out a histogram of TTE as a csv
        fig1, ax = plt.subplots(2)
        quant, bin_loc = np.histogram(cuts[disc_y_t],bins=surv.shape[1])
        ax[1].bar(bin_loc[1:],quant,width= (max(cuts)-min(cuts))/len(cuts))
        ax[1].set(xlabel = 'Time to event or censor (years)' , ylabel = 'Sample Count' )
        plot_file_path = os.path.join(temp_path, 'Time Dist Histogram.pdf')
        fig1.savefig(plot_file_path)
        
        hist_path = os.path.join(temp_path, 'Histogram.csv')
        temp = np.transpose(np.vstack( (bin_loc[1:], quant, quant[0] + sum(quant) - np.cumsum(quant))))
        headers = "bin end time, quantity in bin, quantity at risk "
        np.savetxt(hist_path, temp, header=headers, delimiter = ',')
        
        print('Finished evaluation. Total time elapsed: ' + str(time.time()-start_time) )
        
        
    #%% Test?
if __name__ == '__main__':
   main()