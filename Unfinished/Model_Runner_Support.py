# -*- coding: utf-8 -*-
"""
This file contains generic functions that Model_Runner_[survival type] call
"""
import os
import h5py
import time
import numpy as np

import json # saving args

from pycox.utils import kaplan_meier # making KM curves
import matplotlib.pyplot as plt # plotting things and saving them

from pycox.evaluation import EvalSurv # concordance and such
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from MODELS.Support_Functions import Data_Split_Rand
from MODELS.Support_Functions import Save_to_hdf5

# %% Arg processing
def get_covariates(args):
    val_covariate_col_list = []
    if ('val_covariate_col_list' in args.keys()):
        val_covariate_col_list = [int(k) for k in args['val_covariate_col_list'][1:-1].split(',')]
        
    test_covariate_col_list = []
    if ('test_covariate_col_list' in args.keys()):
        test_covariate_col_list = [int(k) for k in args['test_covariate_col_list'][1:-1].split(',')]
        
    return val_covariate_col_list, test_covariate_col_list




# %% Load Data
def Load_Data(args):
    start_time = time.time()
    Data = {}
    datapath1 = os.path.dirname(os.getcwd()) # cleverly jump one one folder without referencing \\ (windows) or '/' (E3)
    with h5py.File(os.path.join(datapath1,'HDF5_DATA',args['Train_Folder'],'TRAIN_DATA','Train_Data.hdf5'), "r") as f:
        Data['x_train'] = f['x'][()]
        Data['y_train'] = f['y'][()]
        Train_Col_Names = [k.decode('UTF-8') for k in f['column_names'][()]]
    
    with h5py.File(os.path.join(datapath1,'HDF5_DATA',args['Test_Folder'],'TEST_DATA','Test_Data.hdf5'), "r") as f:
        Data['x_test'] = f['x'][()]
        Data['y_test'] = f['y'][()] 
        Test_Col_Names = [k.decode('UTF-8') for k in f['column_names'][()]]
        
    print('Model_Runner: Loaded Train and Test data. Data Load Time: ' + str(time.time()-start_time) )
    
    # check if PID is processed correctly
    if (max(Data['y_train'][:,0]) > 16777217):
        if (Data['y_train'].dtype == np.float32):
            print('PID exceeds float32 limits!')
            assert(Data['y_train'].dtype != np.float32)
    
    return Data, Train_Col_Names, Test_Col_Names

# %% Clean Data
def Clean_Data(Data, args):
    # Input: Dictionary Data with keys 'x_train', 'y_train', 'x_test', 'y_test'. Each is a numpy array
    # Output: None. Dictionaries are passed by reference, so we change the arrays idrectly
    # Task: Clean the data.
    # Sometimes time-to-event is '-1.0' meaning no follow up time.
    # Sometimes TTE < 0 (recording error?)
    # Sometimes ECG (x_train/test) contains NaNs
    # Find and trash those indices  
    
    start_time = time.time()
    for key in ['y_train', 'y_test']:
        if ( (key in Data.keys()) and (len(Data[key].shape) > 1) ):
            x_key = 'x' + key[1:]
            
            # mark negative TTE
            neg_inds = np.where(Data[key][:,int(args['y_col_train_time'])] < 0)[0]
            inds_to_del = neg_inds.tolist()
            
            # mark nan traces (5x faster than summing isnan over the whole array)
            for i in range(Data[x_key].shape[0]):
                if (np.isnan(Data[x_key][i]).any()):
                    if i not in inds_to_del:
                        inds_to_del.append(i)           
                     
            # remove data, avoid calling np.delete on Data cause that doubles RAM - just select the indices to keep instead
            if (len(inds_to_del) > 0):
                print('removing ' + str(len(inds_to_del)) + ' inds with nan or negative time')
                inds_to_keep = np.delete( np.arange(Data[key].shape[0]), inds_to_del )
                Data[x_key] = Data[x_key][inds_to_keep] 
                Data[key] = Data[key][inds_to_keep] 
            
    # Don't return anything - Data is a dict and so passed by reff
    print('Model_Runner: Checked data for negative time and nan ECG. Data Clean Time: ' + str(time.time()-start_time) )

# %% Horizoning
# Classifiers have to compact event (1/0) and time-to-event (flt > 0) into a single value (because event->1 as TTE-> 120)
# This requires picking a time horizon and somehow re-labeling the data or throwing some out
# Here, we declare that an event with TTE < 'horizon' is a '1', otherwise a '0'.
# This means that patients that are censored ARE ASSUMED TO HAVE SURVIVED.
# ... which is imperfect but isn't particularly unreasonable for Code-15 or MIMIC-IV

def Apply_Horizon(Data, args):
    # Meant for classifiers 
    # find which cases have events preceding the 'horizon' arg
    # marks those as a '1', else 0. [E*]
    # appends 'y_data' (the labels) with [PID, TTE*, E*] at indices [-3, -2, -1], respectively. 
    # ^ TTE* = TTE. That's there in case we want to add right-censoring, which we currently aren't.
    start_time = time.time()

    assert('horizon' in args.keys())
    assert('y_col_train_time' in args.keys())
    assert('y_col_train_event' in args.keys())
    assert('y_col_test_time' in args.keys())
    assert('y_col_test_event' in args.keys())
    
    # y_train
    # expand Data['y_train'] with extra columns showing PID, TTE*, and E*, where TTE* and E* are the times/events we are training on. Don't modify original TTE/E, which we evaluate survival models on.
    assert ( ('y_train' in Data.keys()) and (len(Data['y_train'].shape) > 1) )
    print('Model_Runner: limiting training TTE to time horizon! assuming class 1 if event by H, else 0 (incl censor)')
    
    horizon = float(args['horizon'])
    times_below_h = Data['y_train'][:,int(args['y_col_train_time'])] <= horizon
    event = abs (Data['y_train'][:,int(args['y_col_train_event'])] -1) < 1e-4
    event_mod = (times_below_h*event).astype(int) # did event happen? Leave a (k,) shape array
    
    # expand y_train - append PID, TTE*, E* @ column inds [-3,-2,-1], respectively
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,0],1)), axis=1)
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,int(args['y_col_train_time'])],1)), axis=1)
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(event_mod,1)), axis=1)
    
        
    # y_test
    assert ( ('y_test' in Data.keys()) and (len(Data['y_test'].shape) > 1) )
    print('Model_Runner: limiting test TTE to time horizon! assuming class 1 if event by H, else 0 (incl censor)')
    
    horizon = float(args['horizon'])
    times_below_h = Data['y_test'][:,int(args['y_col_test_time'])] <= horizon
    event = abs (Data['y_test'][:,int(args['y_col_test_event'])] -1) < 1e-4
    event_mod = (times_below_h*event).astype(int) # did event happen? Leave a (k,) shape array
    
    # expand y_test - append PID, TTE*, E* @ column inds [-3,-2,-1], respectively
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,0],1)), axis=1)
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,int(args['y_col_test_time'])],1)), axis=1)
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(event_mod,1)), axis=1)
    
    print('Model_Runner: Computed E* for Horizon and augmented Data[y_]. Time Taken: ' + str(time.time()-start_time) )

# %% PyCox Horizoning-step equivalent (doesn't horizon):
def Augment_Y(Data, args):
    start_time = time.time()
    
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,0],1)), axis=1) # PID is assumed to be column 0
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,int(args['y_col_train_time'])],1)), axis=1)
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,int(args['y_col_train_event'])],1)), axis=1)

    # expand y_test - append PID, TTE*, E* @ column inds [-3,-2,-1], respectively
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,0],1)), axis=1)
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,int(args['y_col_test_time'])],1)), axis=1)
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,int(args['y_col_test_event'])],1)), axis=1)
    
    print('Model_Runner: Restructured Data. Total time elapsed: ' + str(time.time()-start_time) ) 

# %% Split Ddata
def Split_Data(Data):
    # Split loaded "training" data RANDOMLY BY PATIENT ID
    # Input is Dict; split will be based on 'y_train'
    # Ooutput: None; working with dict passed by reference
    start_time = time.time()
    
    if ('y_train' in Data.keys()):
        
        # NeurIPS version - split Training dataset 80 / 20 into Tr/Val. Test is separate file.
        TR = 80
        VA = 20
        TE = 00
        
        # Per ID, find matching data rows      
        Subj_IDs = Data['y_train'][:,-3]        # PID is now at -3    
        Subj_IDs_Unique = np.unique(Subj_IDs)
        
        #Speedup 08/14/24. Much faster than repeat calls to np.where()
        Subj_ID_to_Rows_Dict = {} # Per PID find rows in the numpy array
        for ind,val in enumerate(Subj_IDs):
            if val in Subj_ID_to_Rows_Dict.keys():
                Subj_ID_to_Rows_Dict[val].append(ind)
            else:
                Subj_ID_to_Rows_Dict[val] = [ind]
                
        # Split the indices of Subj_IDs_Unique into train/validation/test
        Train_Inds, Val_Inds, Test_Inds = Data_Split_Rand( [k for k in range(len(Subj_IDs_Unique))], TR, VA, TE)

        # Speedup 08/14/24
        Train_Inds_ECG  = [Row for Unique_PID_Ind in Train_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[Unique_PID_Ind]] ]
        Val_Inds_ECG  = [Row for Unique_PID_Ind in Val_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[Unique_PID_Ind]] ]
        # Test_Inds_ECG = [Row for PID in Test_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        
        Data['x_valid'] = Data['x_train'][Val_Inds_ECG]
        Data['x_train'] = Data['x_train'][Train_Inds_ECG]
    
        Data['y_valid'] = Data['y_train'][Val_Inds_ECG]
        Data['y_train'] = Data['y_train'][Train_Inds_ECG]
          
    else:
        print('No Data Split')
    
    print('Model_Runner: Split Train into Train/Valid. Data Split Time: ' + str(time.time()-start_time) )                            
        
# %% Debug Data subset - pick 1k elements of train/val/test
def DebugSubset_Data(Data, args):
    
    if args['debug'] == 'True':
        debug = True
        sub_len = 1000
        if (debug):
            print("Model_Runner: WARNING - DEBUG speedup! only using "+str(sub_len)+' elems of tr/val/test!')
            
            tr_inds = np.random.randint(0, Data['x_train'].shape[0], (sub_len))
            va_inds = np.random.randint(0, Data['x_valid'].shape[0], (sub_len))
            te_inds = np.random.randint(0, Data['x_test'].shape[0], (sub_len))
            
            Data['x_train'] = Data['x_train'][tr_inds,:]
            Data['x_valid'] = Data['x_valid'][va_inds,:]
            Data['x_test'] = Data['x_test'][te_inds,:]
            
            Data['y_train'] = Data['y_train'][tr_inds]
            Data['y_valid'] = Data['y_valid'][va_inds]
            Data['y_test'] = Data['y_test'][te_inds]
            
# %% Folders            

# Set up Train Folders
def set_up_train_folders(args):
    # input: model args
    # output: none. we make folders and change args, which is a dict so passed by reference
    # trained models live in /trained_models/[train_data_folder]/Resnet18_Bob/asdf.pt
    #    model evals live in /trained_models/[train_data_folder]/Resnet18_Bob/Eval/[Eval_data_folder]/output.csv    
    
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
    
# evaluation folders
def set_up_test_folders(args):
    # input: model args
    # output: none. we make folders
    # trained models live in /trained_models/[train_data_folder]/Resnet18_Bob/asdf.pt
    #    model evals live in /trained_models/[train_data_folder]/Resnet18_Bob/Eval/[Eval_data_folder]/output.csv    
    
    # Set up .../Eval
    temp_path = os.path.join(args['Model_Folder_Path'], 'EVAL')
    if (os.path.isdir(temp_path) == False):
        os.mkdir(temp_path)
        
    # Set up .../Eval/eval_data_folder
    temp_path = os.path.join(temp_path, args['Test_Folder'] + ' Test_Folder')  
    if (os.path.isdir(temp_path) == False):
        os.mkdir(temp_path)
        
    # Save out Evaluation Args
    path = os.path.join(temp_path, 'Eval_Args.txt')
    with open(path, 'w') as file:
         file.write(json.dumps(args)) # use `json.loads` to do the reverse
         
    args['Model_Eval_Path'] = temp_path
    

    

# %% Check that arrays are passed by Dict
def asdf(arrdict):
    #arrdict['a'] = [1,2,3]
    arrdict['a'] = arrdict['a'][1:]
    return arrdict

# %% metrics and wrappers
def get_surv_briercordance(disc_y_t, disc_y_e, surv_df, target_times, time_points):
    # disc_y_t - (N,) int numpy array of discretized times when event occurs
    # disc_y_e - (N,) int or bool numpy array of whether event occurs
    # target_times - float list. Which years we care to sample (will pick nearest neighbor in time_points)
    # time_points - (N,) numpy array of which surv rows correspond to which years
    # 
    # get IPCW brier score and concordance
    # ... only at time_points (years) closest to target_times (years)
    # ... this necessarily requires forced right-censoring at target_times < max (time_points)
    
    # we're requesting performance at times (yr), but we need the time points(index) corresponding to those times
    right_censor_time_point_list = []
    for k in target_times:
        a = np.argmin( abs(time_points - k))
        right_censor_time_point_list.append(a)

    # prep plot-compatible output storage
    ipcw_brier_store_all_ecg  = -1 * np.ones(time_points.shape)
    concordance_store_all_ecg = -1 * np.ones(time_points.shape)
    chance_at_censored_point  = -1 * np.ones(time_points.shape)

    for time_point in right_censor_time_point_list: 
        if time_point == 0: # scores are not defined at '0' 
            continue
        
        # otherwise, right-censor and measure 
        disc_y_t_temp = np.copy(disc_y_t)
        disc_y_e_temp = np.copy(disc_y_e)
        
        temp_censor_inds = np.where(disc_y_t >time_point)[0]
        disc_y_t_temp[temp_censor_inds] = time_point
        disc_y_e_temp[temp_censor_inds] = 0
        
        evsurv = EvalSurv(surv_df, disc_y_t_temp, disc_y_e_temp, censor_surv='km') 
        
        #sometimes nothing happens at earlier time points. 
        if (max(disc_y_t_temp) == min(disc_y_t_temp)):
            temp_c = -1
        else:
            temp_c = evsurv.concordance_td('antolini')

        temp_b = evsurv.integrated_brier_score(np.arange(min(time_point,time_points.shape[0]))) # very unstable beyond the censor point, so stop there
        temp_chance = sum(disc_y_e_temp)/len(disc_y_e_temp) # a guess of the "chance" rate (assumes all censored patients survive)
        
        concordance_store_all_ecg[time_point] = temp_c
        ipcw_brier_store_all_ecg[time_point]  = temp_b
        chance_at_censored_point[time_point]  = temp_chance
        
    return (concordance_store_all_ecg, ipcw_brier_store_all_ecg, chance_at_censored_point)

def get_AUROC_AUPRC(disc_y_t, disc_y_e, surv, target_times, time_points):
    # Measure AUROC and AUPRC at several right-censored locations
    # disc_y_t - (N,) int numpy array of discretized times when event occurs
    # disc_y_e - (N,) int or bool numpy array of whether event occurs
    # target_times - float list. Which years we care to sample (will pick nearest neighbor in time_points)
    # time_points - (N,) numpy array of which surv rows correspond to which years

    right_censor_time_point_list = []
    for k in target_times:
        a = np.argmin( abs(time_points - k))
        right_censor_time_point_list.append(a)
    
    AUROC_store = []
    AUPRC_store = []
    Chance_At_Age = []
    for k in range(surv.shape[1]):
        
        if k not in right_censor_time_point_list:
            AUROC_store.append(-1)
            AUPRC_store.append(-1)
            Chance_At_Age.append(-1)
            continue

        correct = ((disc_y_t<=k) * (disc_y_e==1)).astype(int)
        estimate = 1-surv[:,k]
        
        try:
            AUROC = roc_auc_score(correct,estimate)
        except:
            AUROC = -1 # sometimes the classifier is all 0 or all 1
            
        try:    
            AUPRC = average_precision_score(correct,estimate)
        except:
            AUPRC = -1
            
        AUROC_store.append(AUROC)
        AUPRC_store.append(AUPRC)
        
        Chance_At_Age.append(sum(correct)/len(correct))
        
    return AUROC_store, AUPRC_store, Chance_At_Age

    

def Gen_KM_Bootstraps(surv, disc_y_e, disc_y_t, sample_time_points, args):
    # wrapper. Generates a KM output, saves out figure and .hdf5 with the values
    start_time = time.time()
    
    km_outs = []
    mdl_outs = []
    for k in range(100):
        sample_inds = np.random.randint(0,surv.shape[0],(surv.shape[0]))
        km_out = kaplan_meier(events=disc_y_e[sample_inds], durations=disc_y_t[sample_inds]).to_numpy().tolist()
        mdl_out = np.mean(surv[sample_inds],axis=0)
        
        # sometimes don't get a last index on the KM - extend last value out
        while (len(km_out) < surv.shape[1]):
            km_out.append(km_out[-1])
            
        km_outs.append(km_out)
        mdl_outs.append(mdl_out)
        
    mdl_outs = np.vstack(mdl_outs)
    km_outs = np.vstack(km_outs)
    
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
    plt.plot(sample_time_points, km_median)
    ax.fill_between(sample_time_points, km_int_low, km_int_high, color='b', alpha=.1)
    plt.plot(sample_time_points,mdl_median, color='r')
    ax.fill_between(sample_time_points, mdl_int_low, mdl_int_high, color='r', alpha=.1)
    plt.legend(('KM','KM 2.5-97.5%','Model','Model 2.5-97.5%'))
    plt.xlabel('Years')
    plt.ylabel('Survival')
    plot_file_path = os.path.join(args['Model_Eval_Path'], 'KM vs Model 100xBS Survival Curve.pdf')
    fig1.savefig(plot_file_path)
    
    outputs_hdf5_path = os.path.join(args['Model_Eval_Path'], 'KM_Outputs.hdf5')
    Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
    Save_to_hdf5(outputs_hdf5_path, mdl_median, 'SF_mdl_median')
    Save_to_hdf5(outputs_hdf5_path, mdl_int_low, 'SF_int_low')
    Save_to_hdf5(outputs_hdf5_path, mdl_int_high, 'SF_mdl_int_high')
    Save_to_hdf5(outputs_hdf5_path, km_median, 'SF_km_median')
    Save_to_hdf5(outputs_hdf5_path, km_int_low, 'SF_km_int_low')
    Save_to_hdf5(outputs_hdf5_path, km_int_high, 'SF_km_int_high')
    
    print('Model_Runner: KM w Bootstraps Time: ' + str(time.time()-start_time) )                            
    
def Gen_Concordance_Brier_No_Bootstrap(surv_df, disc_y_t, disc_y_e, time_points, sample_time_points, args):
    # wrapper. Gets concordance and Brier (limited to time-points in time_points, however right that is)
    start_time = time.time()
    
    # discretize test_outputs time
    concordance_store_all_ecg, ipcw_brier_store_all_ecg, chance_at_censored_point_all_ecg  = get_surv_briercordance(disc_y_t, disc_y_e, surv_df, time_points, sample_time_points)
    
    hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
    Save_to_hdf5(hdf5_path, concordance_store_all_ecg, 'concordance_store_all_ecg')
    Save_to_hdf5(hdf5_path, ipcw_brier_store_all_ecg, 'ipcw_brier_store_all_ecg')
    Save_to_hdf5(hdf5_path, chance_at_censored_point_all_ecg, 'chance_at_censored_point_all_ecg')
    
    fig1, ax = plt.subplots()
    plt.plot([0,max(sample_time_points)],[.5,.5], '--') # concordance if guessing
    plt.plot(sample_time_points, ipcw_brier_store_all_ecg)
    plt.plot(sample_time_points, concordance_store_all_ecg)
    plt.ylim([0,1])
    plt.legend(['Concordance Chance','Brier = MSE','Concordance'])
    plt.xlabel('follow-up time (years)')
    plt.ylabel('Measure for time bin')
    plot_file_path = os.path.join(args['Model_Eval_Path'], 'briercordance, briercordance.pdf')
    fig1.savefig(plot_file_path)
    
    print('Model_Runner: No-Bootstrap Conc/Brier Time: ' + str(time.time()-start_time) )    

def Gen_Concordance_Brier_PID_Bootstrap(Data, args, disc_y_t, disc_y_e, surv_df, sample_time_points, time_points):
    # wrapper. Gets concordance and brier (limited to time-points in time_points, however right that is), picking 1 ECG per patient x 20.
    start_time = time.time()
    
    # Picking a random ECG per PID, calc concordance and Brier at time_points
    # (later, they will be combined for all the random seeds into a single mean / stdev / 95% CI)
    # 1. find relevant rows per subjetc ID
    if (args['Eval_Dataloader'] == 'Validation'):
        Subj_IDs = Data['y_valid'][:,-3]    # PID lives in -3
    elif (args['Eval_Dataloader'] == 'Train'):
        Subj_IDs = Data['y_train'][:,-3]    
    else:
        Subj_IDs = Data['y_test'][:,-3]  
        
    Subj_IDs_Unique = np.unique(Subj_IDs)
    Subj_ID_to_Rows_Dict = {} # map ID to rows
    for ind,val in enumerate(Subj_IDs):
        if val in Subj_ID_to_Rows_Dict.keys():
            Subj_ID_to_Rows_Dict[val].append(ind)
        else:
            Subj_ID_to_Rows_Dict[val] = [ind]
        
    bootstrap_briers = [] # list of lists
    bootstrap_concordances = [] # list of lists
    
    bootstraps = 20 
    Inds = [Subj_ID_to_Rows_Dict[k][0] for k in Subj_IDs_Unique]
    
    for b in range (bootstraps):
        # 3. Sample one revelant Surv row per each subject.
        for i,s in enumerate(Subj_IDs_Unique):
            tmp = Subj_ID_to_Rows_Dict[s]
            if (len(tmp) != 1):
                Inds[i] = tmp[np.random.randint(0,len(tmp))]
        
        concordance_score, ipcw_brier_score, chance_at_censored_point  = get_surv_briercordance(disc_y_t[Inds], disc_y_e[Inds], surv_df.iloc[:,Inds], time_points, sample_time_points)
        
        bootstrap_briers.append(ipcw_brier_score)
        bootstrap_concordances.append(concordance_score)
            
    hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
    Save_to_hdf5(hdf5_path, bootstrap_briers, 'bootstrap_briers')
    Save_to_hdf5(hdf5_path, bootstrap_concordances, 'bootstrap_concordances')
    
    print('Model_Runner: Bootstrap Conc/Brier Time: ' + str(time.time()-start_time) )    
    
def Gen_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points, args):
    # Wrapper. Calculates AUROC/AUPRC for each point in time_points.
    # Note: we need to compress TTE/E here again.
    # We declare a case a '1' if death by time in sample_time, else 0 (EVEN IF CENSORED)'
    # This is not perfectly correct, but there doesn't seem to be a good solution
    start_time = time.time()
    
    S_AUROC_store, S_AUPRC_store, Chance_At_Age = get_AUROC_AUPRC(disc_y_t, disc_y_e, surv, time_points, sample_time_points)
    
    fig1, ax = plt.subplots()
    plt.plot([0,max(sample_time_points)],[.5,.5], '--')
    plt.plot(sample_time_points,Chance_At_Age,'--')
    plt.plot(sample_time_points, S_AUROC_store)
    plt.plot(sample_time_points, S_AUPRC_store)
    plt.ylim([0,1])
    plt.legend(['AUROC Chance' , 'Chance at time','AUROC','AUPRC'])
    plt.xlabel('follow-up time (years)')
    plt.ylabel('Measure for time bin')
    plot_file_path = os.path.join(args['Model_Eval_Path'], 'AUROC, AUPRC.pdf')
    fig1.savefig(plot_file_path)
    
    hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
    Save_to_hdf5(hdf5_path, S_AUROC_store, 'AUROC')
    Save_to_hdf5(hdf5_path, S_AUPRC_store, 'AUPRC')
    print('Model_Runner: AUROC/AUPRC Time: ' + str(time.time()-start_time) )    
    
def print_classifier_ROC(test_correct_outputs, test_outputs):
    # how did the classifier perform at its classification task?
    Classif_AUROC = roc_auc_score(test_correct_outputs,test_outputs)
    Classif_AUPRC = average_precision_score(test_correct_outputs,test_outputs)
    print('on the classification task itself')
    print('Classif_AUROC ' + str(Classif_AUROC))
    print('Classif_AUPRC ' + str(Classif_AUPRC))
    
def save_histogram(sample_time_points, disc_y_t, surv, args):
    fig1, ax = plt.subplots(2)
    quant, bin_loc = np.histogram(sample_time_points[disc_y_t],bins=surv.shape[1])
    ax[1].bar(bin_loc[1:],quant,width= (max(sample_time_points)-min(sample_time_points))/len(sample_time_points))
    ax[1].set(xlabel = 'Time to event or censor (years)' , ylabel = 'Sample Count' )
    plot_file_path = os.path.join(args['Model_Eval_Path'], 'Time Dist Histogram.pdf')
    fig1.savefig(plot_file_path)
    
    hist_path = os.path.join(args['Model_Eval_Path'], 'Histogram.csv')
    temp = np.transpose(np.vstack( (bin_loc[1:], quant, quant[0] + sum(quant) - np.cumsum(quant))))
    headers = "bin end time, quantity in bin, quantity at risk "
    np.savetxt(hist_path, temp, header=headers, delimiter = ',')    