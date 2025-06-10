# -*- coding: utf-8 -*-
"""
This file contains generic functions that Model_Runner_[survival type] call
"""
import os
import h5py
import time
import numpy as np
import pandas as pd

import json # saving args

from pycox.utils import kaplan_meier # making KM curves
import matplotlib.pyplot as plt # plotting things and saving them

from pycox.evaluation import EvalSurv # concordance and such
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from MODELS.Support_Functions import Data_Split_Rand
from MODELS.Support_Functions import Save_to_hdf5

# %% compatability for old sk-learn: import simps
#taken directly from  from https://github.com/scipy/scipy/blob/v0.18.1/scipy/integrate/quadrature.py#L298
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def _basic_simps(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even spaced Simpson's rule.
        result = np.sum(dx/3.0 * (y[slice0]+4*y[slice1]+y[slice2]),
                        axis=axis)
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0]
        h1 = h[sl1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1
        tmp = hsum/6.0 * (y[slice0]*(2-1.0/h0divh1) +
                          y[slice1]*hsum*hsum/hprod +
                          y[slice2]*(2-h0divh1))
        result = np.sum(tmp, axis=axis)
    return result

def simps(y, x=None, dx=1, axis=-1, even='avg'):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule.  If x is None, spacing of dx is assumed.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals.  The parameter 'even' controls how this is handled.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : int, optional
        Spacing of integration points along axis of `y`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : {'avg', 'first', 'str'}, optional
        'avg' : Average two results:1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.

        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.

        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

    See Also
    --------
    quad: adaptive quadrature using QUADPACK
    romberg: adaptive Romberg quadrature
    quadrature: adaptive Gaussian quadrature
    fixed_quad: fixed-order Gaussian quadrature
    dblquad: double integrals
    tplquad: triple integrals
    romb: integrators for sampled data
    cumtrapz: cumulative integration for sampled data
    ode: ODE integrators
    odeint: ODE integrators

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less.  If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-d or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice1 = (slice(None),)*nd
        slice2 = (slice(None),)*nd
        if even not in ['avg', 'last', 'first']:
            raise ValueError("Parameter 'even' must be "
                             "'avg', 'last', or 'first'.")
        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5*last_dx*(y[slice1]+y[slice2])
            result = _basic_simps(y, 0, N-3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5*first_dx*(y[slice2]+y[slice1])
            result += _basic_simps(y, 1, N-2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simps(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


# %% Arg processing
# def get_covariates(args):
#     val_covariate_col_list = []
#     if ('val_covariate_col_list' in args.keys()):
#         val_covariate_col_list = [int(k) for k in args['val_covariate_col_list'][1:-1].split(',')]
        
#     test_covariate_col_list = []
#     if ('test_covariate_col_list' in args.keys()):
#         test_covariate_col_list = [int(k) for k in args['test_covariate_col_list'][1:-1].split(',')]
        
#     return val_covariate_col_list, test_covariate_col_list


# def provide_data_details(args, Data, Train_Col_Names, Test_Col_Names):
#     from scipy.stats import ttest_ind
#     from scipy.stats import chi2_contingency
#     # after the data has been cleaned, we can look at some distributions
    
#     # only get data details from the training set
#     if (Train_Col_Names !=Test_Col_Names):
#         return 0   

#     PID_index = 0 # PID is always at zero
#     event_index = int(args['y_col_train_event'])
#     tte_index = int(args['y_col_train_time'])

#     # find 'is_male' column ('is_male' in BCH, Code-15, 'Gender' in MIMIC)
#     for i,k in enumerate(Train_Col_Names):
#         if k == 'Gender' or k== 'is_male':
#             is_male_index=i
#             break
    
#     # find 'age' column ('age' in BCH, Code-15, 'Age' in MIMIC)
#     for i,k in enumerate(Train_Col_Names):
#         if k == 'age' or k== 'Age':
#             age_index=i
#             break
        
#     train_cases = Data['y_train']
#     test_cases = Data['y_test']
#     combined_cases = np.vstack((Data['y_train'], Data['y_test']))
    
#     male_cases =   combined_cases[combined_cases[:,is_male_index] == 1]
#     female_cases = combined_cases[combined_cases[:,is_male_index] == 0]
    
#     male_pos_cases = male_cases[male_cases[:,event_index]==1,:]
#     male_neg_cases = male_cases[male_cases[:,event_index]==0,:]
    
#     female_pos_cases = female_cases[female_cases[:,event_index]==1,:]
#     female_neg_cases = female_cases[female_cases[:,event_index]==0,:]
    
    
#     pos_cases = combined_cases[combined_cases[:,event_index]==1]
#     neg_cases = combined_cases[combined_cases[:,event_index]==0]
    
#     print('\n by ECG')
#     print('ECG_n ', len(train_cases)+len(test_cases))
#     print('ECG_n Pos', sum(train_cases[:,event_index]==1)  + sum(test_cases[:,event_index]==1) )
#     print('ECG_n Neg', sum(train_cases[:,event_index]==0)  + sum(test_cases[:,event_index]==0) )
#     print('Do these add up? ', (sum(train_cases[:,event_index]==1)  + sum(test_cases[:,event_index]==1) + sum(train_cases[:,event_index]==0)  + sum(test_cases[:,event_index]==0)  == len(train_cases)+len(test_cases)))
    
#     print('\n by PID')
#     print('train PID unique ', len(np.unique(Data['y_train'][:,PID_index])))
#     print('test PID unique ', len(np.unique(Data['y_test'][:,PID_index])))
#     print('test+train PID unique ', len(np.unique(combined_cases[:,PID_index])))
#     print('Do these add up? ', len(np.unique(Data['y_test'][:,PID_index])) + len(np.unique(Data['y_train'][:,PID_index])) == len(np.unique(combined_cases[:,PID_index])))
    
#     uniques,unique_ind = np.unique(combined_cases[:,PID_index], return_index=True)
#     print('PID pos', sum(combined_cases[unique_ind,event_index]==1))
#     print('PID neg', sum(combined_cases[unique_ind,event_index]==0))
#     print('Do these line up? ', sum(combined_cases[unique_ind,event_index]==1) + sum(combined_cases[unique_ind,event_index]==0) == len(np.unique(combined_cases[:,PID_index])))
    
#     # age mean, stdev, by gender
#     print ('\n age (stdev)')
#     print('age ... all ', np.mean(combined_cases[:,age_index]), np.std(combined_cases[:,age_index]) )
#     print('age ... pos ', np.mean(pos_cases[:,age_index]), np.std(pos_cases[:,age_index]) )
#     print('age ... neg ', np.mean(neg_cases[:,age_index]), np.std(neg_cases[:,age_index]) )
#     print('unpared t test: age ~ pos/neg:')
#     print(ttest_ind (pos_cases[:,age_index],neg_cases[:,age_index]))
    
#     # by sex,
#     print('\n F/M')
#     print('Num female cases, % of all',len(female_cases), len(female_cases) / (len(female_cases) + len(male_cases)))
#     print('Pos female cases, % of pos',len(female_pos_cases), len(female_pos_cases) / (len(female_pos_cases) + len(male_pos_cases)))
#     print('Neg female cases, % of neg',len(female_neg_cases), len(female_neg_cases) / (len(female_neg_cases) + len(male_neg_cases)))
    
#     print('Num male cases, % of all',len(male_cases), len(male_cases) / (len(female_cases) + len(male_cases)))
#     print('Pos male cases, % of pos',len(male_pos_cases), len(male_pos_cases) / (len(female_pos_cases) + len(male_pos_cases)))
#     print('Neg male cases, % of neg',len(male_neg_cases), len(male_neg_cases) / (len(female_neg_cases) + len(male_neg_cases)))
#     print('chi square test on contingency table: gender ~ pos/neg:')
#     print(chi2_contingency( [[len(female_pos_cases), len(female_neg_cases)], [len(male_pos_cases), len(male_neg_cases)]]))
    
#     # followup years
#     print('\n Followup')
#     print('followup all ', np.mean(combined_cases[:,tte_index]), np.std(combined_cases[:,tte_index]))
#     print('followup pos ', np.mean(pos_cases[:,tte_index]), np.std(pos_cases[:,tte_index]))
#     print('followup neg ', np.mean(neg_cases[:,tte_index]), np.std(neg_cases[:,tte_index]))
#     print('unpared t test: followup T ~ pos/neg:')
#     print(ttest_ind (pos_cases[:,tte_index],neg_cases[:,tte_index]))

# %% Load Data
def Load_Labels(args):
    start_time = time.time()
    datapath1 = os.path.dirname(os.getcwd()) # cleverly jump one one folder without referencing \\ (windows) or '/' (E3)

    # Train: pull in one dataset or all?
    if (args['Train_Folder'] == 'All'):
        train_csv_path_C = os.path.join(datapath1, 'HDF5_DATA','Code15','Labels_Code15_mort_032025_pd_8020.csv')
        train_csv_path_B = os.path.join(datapath1, 'HDF5_DATA','BCH','BCH_Mort_Labels_042225.csv')
        train_csv_path_M = os.path.join(datapath1, 'HDF5_DATA','MIMICIV','Labels_MIMICIV_mort_032025_pd_8020.csv')
        
        # pull datasets
        train_labels_C = pd.read_csv(train_csv_path_C)
        train_labels_B = pd.read_csv(train_csv_path_B)
        train_labels_M = pd.read_csv(train_csv_path_M)
        
        train_labels_C['Dataset'] = 'Code15'
        train_labels_B['Dataset'] = 'BCH'
        train_labels_M['Dataset'] = 'MIMICIV'
        
        # merge datasets
        train_labels = pd.concat((train_labels_C,train_labels_B,train_labels_M)).reset_index(drop=True) # without reset, index = 0,0,0,1,1,1,etc.
        train_rows = train_labels[train_labels['Test_Train_split_12345']=='tr'].index.values

    # Just one
    else:
        if (args['Train_Folder'] == 'Code15'):
            train_csv_path = os.path.join(datapath1, 'HDF5_DATA',args['Train_Folder'],'Labels_Code15_mort_032025_pd_8020.csv')
        if (args['Train_Folder'] == 'BCH'):
            train_csv_path = os.path.join(datapath1, 'HDF5_DATA',args['Train_Folder'],'BCH_Mort_Labels_042225.csv')
        if (args['Train_Folder'] == 'MIMICIV'):
            train_csv_path = os.path.join(datapath1, 'HDF5_DATA',args['Train_Folder'],'Labels_MIMICIV_mort_032025_pd_8020.csv')
            
        train_labels = pd.read_csv(train_csv_path)
        train_rows = train_labels[train_labels['Test_Train_split_12345']=='tr'].index.values

    # Pull Test Data
    if (args['Test_Folder'] == 'Code15'):
        test_csv_path = os.path.join(datapath1, 'HDF5_DATA',args['Test_Folder'],'Labels_Code15_mort_032025_pd_8020.csv')
    if (args['Test_Folder'] == 'BCH'):
        test_csv_path = os.path.join(datapath1, 'HDF5_DATA',args['Test_Folder'],'BCH_Mort_Labels_042225.csv')
    if (args['Test_Folder'] == 'MIMICIV'):
        test_csv_path = os.path.join(datapath1, 'HDF5_DATA',args['Test_Folder'],'Labels_MIMICIV_mort_032025_pd_8020.csv')
        
    test_labels = pd.read_csv(test_csv_path)
    test_rows  = test_labels[test_labels['Test_Train_split_12345']=='te'].index.values
    
    # 3. split labels into test/train
    train_df  = train_labels.loc[train_rows].copy().reset_index(drop=True)
    test_df   = test_labels.loc[test_rows].copy().reset_index(drop=True)
        
    # 5. check if PID is processed correctly
    if ((max(train_labels['PID']) > 16777217) or ((max(test_labels['PID']) > 16777217))):
        if ((train_labels['PID'].dtype == 'float32') or (test_labels['PID'].dtype == 'float32')):
            print('PID exceeds float32 limits, but is float32!')
            breakpoint()
            
    return  train_df, test_df
# %%
def Load_ECG_and_Cov(train_df, valid_df, test_df, args):
    # Handle debugging: limit df if needed
    # Load ECG; match the SID in case the ECG is out of order 
    # Input: dataframes, args
    # Outputs: Data{}, containing ECG and covariates in numpy arrays
    
    start_time = time.time()
    Data={}
    datapath1 = os.path.dirname(os.getcwd()) # cleverly jump one one folder without referencing \\ (windows) or '/' (E3)
    
    # If pulling all training data
    if (args['Train_Folder'] == 'All'):

        # Split train/valid df by source (not needed for test set: one at a time there)
        train_df_C = train_df[train_df['Dataset']=='Code15']
        train_df_B = train_df[train_df['Dataset']=='BCH']
        train_df_M = train_df[train_df['Dataset']=='MIMICIV']
        
        valid_df_C = valid_df[valid_df['Dataset']=='Code15']
        valid_df_B = valid_df[valid_df['Dataset']=='BCH']
        valid_df_M = valid_df[valid_df['Dataset']=='MIMICIV']
        
        # Handle Debug Case
        if ('debug' in args.keys()):
            if (args['debug']=='True'): # keep 5K Tr/VAl/test each (do here, was NOT done earlier)
                test_df = test_df.loc[0:4999].copy().reset_index(drop=True)
                
                train_df_C = train_df_C.loc[0:4999].copy().reset_index(drop=True)
                train_df_B = train_df_B.loc[0:4999].copy().reset_index(drop=True)
                train_df_M = train_df_M.loc[0:4999].copy().reset_index(drop=True)
                
                valid_df_C = valid_df_C.loc[0:4999].copy().reset_index(drop=True)
                valid_df_B = valid_df_B.loc[0:4999].copy().reset_index(drop=True)
                valid_df_M = valid_df_M.loc[0:4999].copy().reset_index(drop=True)
                
        # FIle locations
        train_ecg_path_C = os.path.join(datapath1, 'HDF5_DATA','Code15','Code15_ECG.hdf5')
        train_ecg_path_B = os.path.join(datapath1, 'HDF5_DATA','BCH','BCH_All_ECG_042225.hdf5')
        train_ecg_path_M = os.path.join(datapath1, 'HDF5_DATA','MIMICIV','MIMIC_ECG.hdf5')
                
        # open hdf5
        train_ECG = []
        valid_ECG = []

        # Pull C15
        with h5py.File(train_ecg_path_C, "r") as f:
            
            # Pull Training ECG data
            ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in train_df_C['SID']]) # Find the rows for our dataframe
            
            train_ECG.append (f['ECG'][np.sort(rows_to_pull)]) # pull those rows in ascending order (hp5y requirement)
            train_df_C['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            train_df_C = train_df_C.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
            # Pull Validation ECG data
            # ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in valid_df_C['SID']]) # Find the rows for our dataframe
            valid_ECG.append(f['ECG'][np.sort(rows_to_pull)]) # pull those rows in ascending order (hp5y requirement)
            valid_df_C['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            valid_df_C = valid_df_C.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
        # Pull BCH
        with h5py.File(train_ecg_path_B, "r") as f:
            
            # Pull Training ECG data
            ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in train_df_B['SID']]) # Find the rows for our dataframe
            
            train_ECG.append (f['ECG'][np.sort(rows_to_pull)]) # pull those rows in ascending order (hp5y requirement)
            train_df_B['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            train_df_B = train_df_B.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
            # Pull Validation ECG data
            # ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in valid_df_B['SID']]) # Find the rows for our dataframe
            valid_ECG.append(f['ECG'][np.sort(rows_to_pull)]) # pull those rows in ascending order (hp5y requirement)
            valid_df_B['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            valid_df_B = valid_df_B.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
        # Pull MIMICIV
        with h5py.File(train_ecg_path_M, "r") as f:
            # Pull Training ECG data
            ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in train_df_M['SID']]) # Find the rows for our dataframe
            
            # swap the MIMICIV ECG rows here to align with C15
            tmp = f['ECG'][np.sort(rows_to_pull)]
            tmp[:,:,[4,5]] = tmp[:,:,[5,4]] # switch ECG order
            train_ECG.append(tmp.copy())
            del tmp
            train_df_M['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            train_df_M = train_df_M.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
            # Pull Validation ECG data
            # ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in valid_df_M['SID']]) # Find the rows for our dataframe
            
            # swap the MIMICIV ECG rows here to align with C15
            tmp = f['ECG'][np.sort(rows_to_pull)]
            tmp[:,:,[4,5]] = tmp[:,:,[5,4]] # switch ECG order
            valid_ECG.append(tmp.copy())
            del tmp
            
            valid_df_M['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            valid_df_M = valid_df_M.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
            # sanity check: 
            # a = valid_ECG[-1][-1]
            # plt.plot( a[:,0]-a[:,1]/2 - a[:,4]) # aVL, [:,4] in C15, is given by I -II/2; this should be roughly 0
            
        # merge the ECG and re-ordered dataframes, del list reference; garbage collector should handle list elements
        train_df = pd.concat((train_df_C, train_df_B, train_df_M))
        valid_df = pd.concat((valid_df_C, valid_df_B, valid_df_M))
        Data['ECG_train'] = np.vstack(train_ECG).astype(np.float32)
        Data['ECG_valid'] = np.vstack(valid_ECG).astype(np.float32)
        del train_ECG
        del valid_ECG
        
    # if pulling just one dataset
    else:
        
        # handle debug case
        if ('debug' in args.keys()):
            if (args['debug']=='True'):
                train_df = train_df.loc[0:4999].copy().reset_index(drop=True)
                valid_df = valid_df.loc[0:4999].copy().reset_index(drop=True)
                test_df  = test_df.loc[0:4999].copy().reset_index(drop=True)
    
        # Find pulling location
        if (args['Train_Folder'] == 'Code15'):
            train_ecg_path = os.path.join(datapath1, 'HDF5_DATA',args['Train_Folder'],'Code15_ECG.hdf5')
        if (args['Train_Folder'] == 'BCH'):
            train_ecg_path = os.path.join(datapath1, 'HDF5_DATA',args['Train_Folder'],'BCH_All_ECG_042225.hdf5')
        if (args['Train_Folder'] == 'MIMICIV'):
            train_ecg_path = os.path.join(datapath1, 'HDF5_DATA',args['Train_Folder'],'MIMIC_ECG.hdf5')
            
        # open hdf5
        with h5py.File(train_ecg_path, "r") as f:
            
            # Pull Training ECG data
            ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in train_df['SID']]) # Find the rows for our dataframe
            Data['ECG_train'] = f['ECG'][np.sort(rows_to_pull)] # pull those rows in ascending order (hp5y requirement)
            train_df['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            train_df = train_df.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
            
            # Pull Validation ECG data
            # ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
            rows_to_pull = np.array([ECG_SID_Row[SID] for SID in valid_df['SID']]) # Find the rows for our dataframe
            Data['ECG_valid'] = f['ECG'][np.sort(rows_to_pull)] # pull those rows in ascending order (hp5y requirement)
            valid_df['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
            valid_df = valid_df.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
        
        
    # Test
    if (args['Test_Folder'] == 'Code15'):
        test_ecg_path = os.path.join(datapath1, 'HDF5_DATA',args['Test_Folder'],'Code15_ECG.hdf5')
    if (args['Test_Folder'] == 'BCH'):
        test_ecg_path = os.path.join(datapath1, 'HDF5_DATA',args['Test_Folder'],'BCH_All_ECG_042225.hdf5')
    if (args['Test_Folder'] == 'MIMICIV'):
        test_ecg_path = os.path.join(datapath1, 'HDF5_DATA',args['Test_Folder'],'MIMIC_ECG.hdf5')

    with h5py.File(test_ecg_path, "r") as f:                            
        # Pull Test ECG data
        ECG_SID_Row  = {SID:i for i,SID in enumerate(f['SID'])} # Which ECG.h5 row corresponds to which SID?
        rows_to_pull = np.array([ECG_SID_Row[SID] for SID in test_df['SID']]) # Find the rows for our dataframe
        Data['ECG_test'] = f['ECG'][np.sort(rows_to_pull)] # pull those rows in ascending order (hp5y requirement)
        test_df['ECG_Row_Num'] = rows_to_pull # sort the dataframe in ascending row-pulled order
        test_df = test_df.sort_values(by=["ECG_Row_Num"]).reset_index(drop=True)
        
    # 5. Assemble Covariates into numpy tables
    if ('covariates' in args.keys()):
        cov_list = args['covariates'][1:-1].split(',')
        Data['Cov_train'] = train_df[cov_list].to_numpy().astype(np.float32)
        Data['Cov_valid'] = valid_df[cov_list].to_numpy().astype(np.float32)
        Data['Cov_test']  =  test_df[cov_list].to_numpy().astype(np.float32)
    else:
        Data['Cov_train'] = []
        Data['Cov_valid'] = []
        Data['Cov_test']  = []
    
    print('Model_Runner: Loaded Train and Test data. Data Load Time: ' + str(time.time()-start_time) )
    
    # if the lead order is different from the Code-15 standard, (MIMIC-IV), reorder test data to match training data
    if (args['Train_Folder'] != args['Test_Folder']):
        # MIMICIV in test folder, change {I, II, III, avR, avF, avL} (from header files, differs from publication) to Code15's {I, II, III, avR, avL, avF}
        if (args['Test_Folder'] == 'MIMICIV') :
            Data['ECG_test'][:,:,[4,5]] = Data['ECG_test'][:,:,[5,4]]
            print('\n Reordered MIMICIV Test set to match non-MIMICIV Train Set\n')
        # MIMICIV in train folder, change non-MIMIC {I, II, III, avR, avL, avF} to MIMIC's {I, II, III, avR, avF, avL} (from header files, differs from publication)
        elif (args['Train_Folder'] == 'MIMICIV'):
            Data['ECG_test'][:,:,[4,5]] = Data['ECG_test'][:,:,[5,4]]
            print('\n Reordered Code15-lead-order Test set to match MIMICIV-lead-order Train Set\n')
        
    
    # you adjusted the dataframes, so return those as well
    return Data, train_df, valid_df, test_df
    

# %% Clean Data [obsolete 04/25/25 - now handled in raw -> hdf5 conversion]
# def Clean_Data(Data, args):
#     # Input: Dictionary Data with keys 'x_train', 'y_train', 'x_test', 'y_test'. Each is a numpy array
#     # Output: None. Dictionaries are passed by reference, so we change the arrays idrectly
#     # Task: Clean the data.
#     # Sometimes time-to-event is '-1.0' meaning no follow up time.
#     # Sometimes TTE < 0 (recording error?)
#     # Sometimes ECG (x_train/test) contains NaNs
#     # Find and trash those indices  
    
#     start_time = time.time()
#     for key in ['y_train', 'y_test']:
#         if ( (key in Data.keys()) and (len(Data[key].shape) > 1) ):
#             x_key = 'x' + key[1:]
            
#             # mark negative TTE
#             if (key == 'y_train'):
#                 neg_inds = np.where(Data[key][:,int(args['y_col_train_time'])] < 0)[0]
#             elif (key == 'y_test'):
#                 neg_inds = np.where(Data[key][:,int(args['y_col_test_time'])] < 0)[0]
#             inds_to_del = neg_inds.tolist()
            
#             # mark nan traces (5x faster than summing isnan over the whole array)
#             for i in range(Data[x_key].shape[0]):
#                 if (np.isnan(Data[x_key][i]).any()):
#                     if i not in inds_to_del:
#                         inds_to_del.append(i)           
                     
#             # remove data, avoid calling np.delete on Data cause that doubles RAM - just select the indices to keep instead
#             if (len(inds_to_del) > 0):
#                 print('removing ' + str(len(inds_to_del)) + ' inds with nan or negative time')
#                 inds_to_keep = np.delete( np.arange(Data[key].shape[0]), inds_to_del )
#                 Data[x_key] = Data[x_key][inds_to_keep] 
#                 Data[key] = Data[key][inds_to_keep] 
            
#     # Don't return anything - Data is a dict and so passed by reff
#     print('Model_Runner: Checked data for negative time and nan ECG. Data Clean Time: ' + str(time.time()-start_time) )

# %% Horizoning
# Classifiers have to compact event (1/0) and time-to-event (flt > 0) into a single value (because event->1 as TTE-> 120)
# This requires picking a time horizon and somehow re-labeling the data or throwing some out
# Here, we declare that an event with TTE < 'horizon' is a '1', otherwise a '0'.
# This means that patients that are censored ARE ASSUMED TO HAVE SURVIVED.
# ... which is imperfect but isn't particularly unreasonable for Code-15 or MIMIC-IV

def Apply_Horizon(train_df, test_df, args):

    # Meant for classifiers 
    # find which cases have events preceding the 'horizon' arg
    # marks those as a '1', else 0. [E*]
    # also append [TTE*]; currently = TTE. That's there in case we want to add right-censoring, which we currently aren't.
    start_time = time.time()
    
    horizon = float(args['horizon'])
    train_df['E*'] = (train_df['Mort_TTE']<horizon * train_df['Mort_Event']).astype(int)
    train_df['TTE*'] = train_df['Mort_TTE']
    
    test_df['E*'] = (test_df['Mort_TTE']<horizon * test_df['Mort_Event']).astype(int)
    test_df['TTE*'] = test_df['Mort_TTE']
    
    print('Model_Runner: Computed E* for Horizon and augmented Data[y_]. Time Taken: ' + str(time.time()-start_time) )

# %% PyCox Horizoning-step equivalent (doesn't horizon):
# def Augment_Y(Data, args):
#     start_time = time.time()
    
#     Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,0],1)), axis=1) # PID is assumed to be column 0
#     Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,int(args['y_col_train_time'])],1)), axis=1)
#     Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,int(args['y_col_train_event'])],1)), axis=1)

#     # expand y_test - append PID, TTE*, E* @ column inds [-3,-2,-1], respectively
#     Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,0],1)), axis=1)
#     Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,int(args['y_col_test_time'])],1)), axis=1)
#     Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,int(args['y_col_test_event'])],1)), axis=1)
    
#     print('Model_Runner: Restructured Data. Total time elapsed: ' + str(time.time()-start_time) ) 

# %% Split Ddata
def Split_Data(train_df):
    # Split loaded "training" data RANDOMLY BY PATIENT ID
    # Into train / validation based on the random seed.
    start_time = time.time()
    
    # NeurIPS version - split Training dataset 80 / 20 into Tr/Val. Test is separate file.
    TR = 80
    VA = 20
    TE = 00
    
    # Per ID, find matching data rows      
    Subj_IDs = train_df['PID']          
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
    
    Val_Inds_ECG = np.array(Val_Inds_ECG)
    
    # Split the training dataframes
    valid_df  = train_df.loc[Val_Inds_ECG].copy().reset_index(drop=True)
    train_df  = train_df.loc[Train_Inds_ECG].copy().reset_index(drop=True)
    
    print('Model_Runner: Split Train into Train/Valid. Data Split Time: ' + str(time.time()-start_time) )                            
    return train_df, valid_df 
        
# %% Debug Data subset - pick 1k elements of train/val/test
# def DebugSubset_Data(Data, args):
#     if('debug' in args.keys()):
#         if args['debug'] == 'True':
#             debug = True
#             sub_len = 1000
#             if (debug):
#                 print("Model_Runner: WARNING - DEBUG speedup! only using "+str(sub_len)+' elems of tr/val/test!')
                
#                 tr_inds = np.random.randint(0, Data['x_train'].shape[0], (sub_len))
#                 va_inds = np.random.randint(0, Data['x_valid'].shape[0], (sub_len))
#                 te_inds = np.random.randint(0, Data['x_test'].shape[0], (sub_len))
                
#                 Data['x_train'] = Data['x_train'][tr_inds,:]
#                 Data['x_valid'] = Data['x_valid'][va_inds,:]
#                 Data['x_test'] = Data['x_test'][te_inds,:]
                
#                 Data['y_train'] = Data['y_train'][tr_inds]
#                 Data['y_valid'] = Data['y_valid'][va_inds]
#                 Data['y_test'] = Data['y_test'][te_inds]
            
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

# def Gen_Concordance_Brier_PID_Bootstrap(Data, args, disc_y_t, disc_y_e, surv_df, sample_time_points, time_points):
    
#     breakpoint()
#     # wrapper. Gets concordance and brier (limited to time-points in time_points, however right that is), picking 1 ECG per patient x 20.
#     start_time = time.time()
    
#     # Picking a random ECG per PID, calc concordance and Brier at time_points
#     # (later, they will be combined for all the random seeds into a single mean / stdev / 95% CI)
#     # 1. find relevant rows per subjetc ID
#     if (args['Eval_Dataloader'] == 'Validation'):
#         Subj_IDs = Data['y_valid'][:,-3]    # PID lives in -3
#     elif (args['Eval_Dataloader'] == 'Train'):
#         Subj_IDs = Data['y_train'][:,-3]    
#     else:
#         Subj_IDs = Data['y_test'][:,-3]  
        
#     Subj_IDs_Unique = np.unique(Subj_IDs)
#     Subj_ID_to_Rows_Dict = {} # map ID to rows
#     for ind,val in enumerate(Subj_IDs):
#         if val in Subj_ID_to_Rows_Dict.keys():
#             Subj_ID_to_Rows_Dict[val].append(ind)
#         else:
#             Subj_ID_to_Rows_Dict[val] = [ind]
        
#     bootstrap_briers = [] # list of lists
#     bootstrap_concordances = [] # list of lists
    
#     bootstraps = 20 
#     Inds = [Subj_ID_to_Rows_Dict[k][0] for k in Subj_IDs_Unique]
    
#     for b in range (bootstraps):
#         # 3. Sample one revelant Surv row per each subject.
#         for i,s in enumerate(Subj_IDs_Unique):
#             tmp = Subj_ID_to_Rows_Dict[s]
#             if (len(tmp) != 1):
#                 Inds[i] = tmp[np.random.randint(0,len(tmp))]
        
#         concordance_score, ipcw_brier_score, chance_at_censored_point  = get_surv_briercordance(disc_y_t[Inds], disc_y_e[Inds], surv_df.iloc[:,Inds], time_points, sample_time_points)
        
#         bootstrap_briers.append(ipcw_brier_score)
#         bootstrap_concordances.append(concordance_score)
            
#     hdf5_path = os.path.join(args['Model_Eval_Path'], 'Stored_Model_Output.hdf5')
#     Save_to_hdf5(hdf5_path, bootstrap_briers, 'bootstrap_briers')
#     Save_to_hdf5(hdf5_path, bootstrap_concordances, 'bootstrap_concordances')
    
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