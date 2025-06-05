# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:58:38 2024

@author: CH242985
"""



import os
import numpy as np
import json
import h5py
import pandas as pd
pd.Series.is_monotonic = pd.Series.is_monotonic_increasing

from pycox.evaluation import EvalSurv
import csv

from tqdm import tqdm
import time

from Model_Runner_Support import get_surv_briercordance

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

# %% Compatability - bring back older version of scipy simpson function
import scipy
from MODELS.Support_Functions import simps
scipy.integrate.simps = simps
# %%
# def get_surv_briercordance(disc_y_t, disc_y_e, surv_df, target_times, time_points):
#     # disc_y_t - (N,) int numpy array of discretized times when event occurs
#     # disc_y_e - (N,) int or bool numpy array of whether event occurs
#     # target_times - float list. Which years we care to sample (will pick nearest neighbor in time_points)
#     # time_points - (N,) numpy array of which surv rows correspond to which years
#     # 
#     # get IPCW brier score and concordance
#     # ... only at time_points (years) closest to target_times (years)
#     # ... this necessarily requires forced right-censoring at target_times < max (time_points)
    
#     # we're requesting performance at times (yr), but we need the time points(index) corresponding to those times
#     right_censor_time_point_list = []
#     for k in target_times:
#         a = np.argmin( abs(time_points - k))
#         right_censor_time_point_list.append(a)

#     # prep plot-compatible output storage
#     ipcw_brier_store_all_ecg  = -1 * np.ones(time_points.shape)
#     concordance_store_all_ecg = -1 * np.ones(time_points.shape)
#     chance_at_censored_point  = -1 * np.ones(time_points.shape)

#     for time_point in right_censor_time_point_list: 
#         if time_point == 0: # scores are not defined at '0' 
#             continue
        
#         # otherwise, right-censor and measure 
#         disc_y_t_temp = np.copy(disc_y_t)
#         disc_y_e_temp = np.copy(disc_y_e)
        
#         temp_censor_inds = np.where(disc_y_t >time_point)[0]
#         disc_y_t_temp[temp_censor_inds] = time_point
#         disc_y_e_temp[temp_censor_inds] = 0
        
#         evsurv = EvalSurv(surv_df, disc_y_t_temp, disc_y_e_temp, censor_surv='km') 
        
#         #sometimes nothing happens at earlier time points. 
#         if (max(disc_y_t_temp) == min(disc_y_t_temp)):
#             temp_c = -1
#         else:
#             temp_c = evsurv.concordance_td('antolini')

#         temp_b = evsurv.integrated_brier_score(np.arange(min(time_point,time_points.shape[0]))) # very unstable beyond the censor point, so stop there
#         temp_chance = sum(disc_y_e_temp)/len(disc_y_e_temp) # a guess of the "chance" rate (assumes all censored patients survive)
        
#         concordance_store_all_ecg[time_point] = temp_c
#         ipcw_brier_store_all_ecg[time_point]  = temp_b
#         chance_at_censored_point[time_point]  = temp_chance
        
#     return (concordance_store_all_ecg, ipcw_brier_store_all_ecg, chance_at_censored_point)


my_path = os.getcwd()
Trained_Models_Path = os.path.join(os.getcwd(), 'Trained_Models')

# getting to
# \CNN\Trained_Models\MIMICIV_Multimodal_Subset\RibeiroClass_MM10\EVAL\MIMICIV_Multimodal_Subs

Model_Dict_List = []
Subgroups_Name_List = [] # easier to track here than later

# Per training Data Source
for Train_Source in os.listdir(Trained_Models_Path): #/MIMICIV_Multimodal_Subset
    
    Train_Path = os.path.join(Trained_Models_Path, Train_Source)
    if (os.path.isdir(Train_Path) == False):
        continue
    
    # Per trained model
    for Model_Name in os.listdir(Train_Path):
        
        # Get model path or continue
        Model_Path = os.path.join(Train_Path, Model_Name) #/RibeiroClass_MM10
        if (os.path.isdir(Model_Path) == False):
            continue
        
        Eval_Path = os.path.join(Model_Path, 'EVAL') #/EVAL
        if (os.path.isdir(Eval_Path) == False):
            continue
        
        for Eval_Data_Folder in os.listdir(Eval_Path): 
            
            Eval_Data_Path = os.path.join(Eval_Path, Eval_Data_Folder)#/MIMICIV_Multimodal_Subs
            if (os.path.isdir(Eval_Data_Path) == False):
                continue
            
            if ('Stored_Model_Output.hdf5' in os.listdir(Eval_Data_Path)):
                # okay, we made it to the evaluation output folder and it has what we need to do an analysis per subgroup
                # print(os.listdir(Eval_Data_Path))
                
                json_path = os.path.join(Eval_Data_Path, 'Eval_Args.txt')   
                with open(json_path) as f:
                    tmp = json.load(f)
                    
                # we only care about the top models with demographics.
                if ('pycox_mdl' not in tmp.keys()): # PyCox_mdl
                    continue
                if ('covariates' not in tmp.keys()): # with dem
                    continue
                if (len(tmp['covariates']) > 13): # and not machine measures
                    continue 
                
                
                if (Train_Source == 'BCH'):
                    if (tmp['pycox_mdl'] != 'DeepHit'):
                        continue
                    if (tmp['Model_Type']!='Ribeiro'):
                        continue
                    
                if (Train_Source == 'Code15'):
                     if (tmp['pycox_mdl'] != 'DeepHit'):
                         continue
                     if (tmp['Model_Type']!='Ribeiro'):
                         continue
                     
                if (Train_Source == 'MIMICIV'):
                     if (tmp['pycox_mdl'] != 'DeepHit'):
                         continue
                     if (tmp['Model_Type']!='Ribeiro'):
                         continue

            
                # store name and data folder
                Model_Dict = {}
                Model_Dict['Name'] = Model_Name                                 
                Model_Dict['Type'] = Model_Name .split('_')[0]    
                Model_Dict['Test_Folder'] = Eval_Data_Folder   
                Model_Dict['Train_Folder'] = Train_Source   
                
                # also store covariates to cluster models by
                Model_Dict['cov'] = tmp['covariates']

                
                # Figure out type from pycox model and horizon
                # breakpoint()
                
                # store pycox model
                if ('pycox_mdl' in tmp.keys()): # PyCox_mdl
                    # Model_Dict['pycox_mdl'] = 'None'                               
                # else:
                    if (tmp['pycox_mdl'] == 'CoxPH'):
                        Model_Dict['Task'] = 'DeepSurv'
                    else:
                        Model_Dict['Task'] = tmp['pycox_mdl']
                   
                # store horizon
                if ('horizon' in tmp.keys()):
                    # Model_Dict['horizon'] = 'None'
                # else:
                    # Model_Dict['horizon'] = tmp['horizon']
                    Model_Dict['Task'] = 'Cla-'+str(int(float(tmp['horizon'])))
                    
                
                
                # store random seed
                Model_Dict['Rand_Seed'] = tmp['Rand_Seed']
                
                # Now we pull the file
                h5py_path = os.path.join(Eval_Data_Path, 'Stored_Model_Output.hdf5')
                with h5py.File(h5py_path, "r") as f:
                    print('--\n\n')
                    print(Model_Name)
                    print(Eval_Data_Folder)
                    keys = [key for key in f.keys()]
                    for i,k in enumerate(keys):
                        print(i,k,f[k][()].shape)
                        
                    surv = f['surv'][()]
                    disc_y_t = f['disc_y_t'][()]
                    disc_y_e = f['disc_y_e'][()]
                    disc_y_e_bool = np.array(disc_y_e,dtype=bool)
                    sample_time_points = f['sample_time_points'][()]
                    
                    Ages = f['Age'][()]
                    Is_Male = f['Is_Male'][()]

                    Age_Brackets = [[0, 20], [20, 40], [40, 60], [60,999], [0, 60], [0,999]]
                    
                    # open with full concordance
                    concordance_list, brier_list, chance_list  = get_surv_briercordance(disc_y_t, disc_y_e_bool, pd.DataFrame(np.transpose(surv)) , [999], sample_time_points) # passed array indicates time of interest (in yr.). Function grabs nearest time point
                    subgroup_name = 'Full Data concordance'
                    concordance = concordance_list[-1]
                    Model_Dict[subgroup_name] = concordance
                    if (subgroup_name not in Subgroups_Name_List):
                        Subgroups_Name_List.append(subgroup_name)
                    
                    for Age_Bracket in Age_Brackets:
                        for gender in np.unique(Is_Male): # For both MIMIC and Code-15, '1' is 'male'
                            
                            # find where gender and age match
                            a = time.time()
                            cat_1 = np.where(Is_Male==gender)
                            cat_2 = np.where(Ages>=Age_Bracket[0])
                            cat_3 = np.where(Ages<Age_Bracket[1])
                            
                            inters_1 = np.intersect1d(cat_1,cat_2)
                            inters_2 = np.intersect1d(inters_1, cat_3)
                            
                            tmp_disc_y_t = disc_y_t[inters_2]
                            tmp_disc_y_e_bool = disc_y_e_bool[inters_2]
                            tmp_surv = surv[inters_2]
                            
                            tmp_surv_df = pd.DataFrame(np.transpose(tmp_surv)) 
                            
                            
                            b = time.time()

                            concordance_list, brier_list, chance_list  = get_surv_briercordance(tmp_disc_y_t, tmp_disc_y_e_bool, tmp_surv_df, [999], sample_time_points) # passed array indicates time of interest (in yr.). Function grabs nearest time point
                            c=time.time()
                            
                            print(Age_Bracket, gender, len(inters_2),'bracket T', '{:.2f}'.format(b-a), 'calc T', '{:.2f}'.format(c-b), 'N event:', sum(tmp_disc_y_e_bool), 'N bracket:', len(tmp_disc_y_e_bool))
                            
                            
                            concordance = concordance_list[-1]
                            brier = brier_list[-1]
                            chance = chance_list[-1]
                            
                            subgroup_name = str(gender) + ',' + str(Age_Bracket) +' concordance'
                            Model_Dict[subgroup_name] = concordance
                            if (subgroup_name not in Subgroups_Name_List):
                                Subgroups_Name_List.append(subgroup_name)
                            
                            # subgroup_name = str(gender) + ',' + str(Age_Bracket) +' brier'
                            # Model_Dict[subgroup_name] = brier
                            # if (subgroup_name not in Subgroups_Name_List):
                            #     Subgroups_Name_List.append(subgroup_name)
                            
                            # subgroup_name = str(gender) + ',' + str(Age_Bracket) +' allT_chance'
                            # Model_Dict[subgroup_name] = chance
                            # if (subgroup_name not in Subgroups_Name_List):
                            #     Subgroups_Name_List.append(subgroup_name)

                    
                Model_Dict_List.append(Model_Dict)
                
# %% okay, we have it sorted. kinda.
# Now we cluster models by matching type, test folder, (train is out for now), and subgroup
inds_clustered = [] # track which dicts have been clustered
clusters = [] # list of lists of inds
cluster_names = [] # list of names (which happen to be lists)

for ind, entry in enumerate(Model_Dict_List):
    if (ind in inds_clustered): # skip if already clustered
        continue
    
    cluster_criteria = ['Type', 'Task', 'Train_Folder', 'Test_Folder','cov']
    
    cluster_name = [entry[k] for k in cluster_criteria]
    cluster_inds = [ind]
    inds_clustered.append(ind) # mark entry1 as clustered
    
    for ind2, entry2 in enumerate(Model_Dict_List):
        if (ind2 in inds_clustered): # don't repeat clustered inds
            continue
        
        # check agreement between the two entries
        move_on = 0
        for crit in cluster_criteria:
            if entry[crit] != entry2[crit]:
                move_on = 1
                break
        if(move_on):
            continue
        
        # the two entries should be in one cluster
        cluster_inds.append(ind2)
        inds_clustered.append(ind2) # mark entry2 as clustered
        
    clusters.append(cluster_inds)
    cluster_names.append(cluster_name)
    
# %% Now per cluster we get the concordance percentiles
def Get_med_25_75_percentile_per_col(Data):
    # input: one dim array
    # output: [median, 25-75th percentile] in str format.
    med = np.median(Data)
    p25 = np.percentile(Data,25)
    p75 = np.percentile(Data,75)
    out_list = [str(np.round(med,decimals=2)), "({:.2f}".format(p25) + ", {:.2f})".format(p75)]
    return out_list

def Get_med_25_75_excel(Data):
    # input: one dim array
    # output: [median, 25-75th percentile] in str format.
    med = np.median(Data)
    p25 = np.percentile(Data,25)
    p75 = np.percentile(Data,75)
    out_list = [str(med), str(med - p25), str(p75 - med)]
    return out_list
    
subgroup_headers = []
for k in Subgroups_Name_List:
    subgroup_headers = subgroup_headers + [k]
    subgroup_headers = subgroup_headers + ['25']
    subgroup_headers = subgroup_headers + ['75']

header_list = cluster_criteria + ['N'] + subgroup_headers
output_list = [header_list]
for cluster_inds, cluster_name in zip(clusters, cluster_names):
    
    cluster_txt_out = [str(len(cluster_inds))] # text output for the cluster
    for subgroup in Subgroups_Name_List:
        
        measures = [ Model_Dict_List[ind][subgroup] for ind in cluster_inds ]
        # txt_out = Get_med_25_75_percentile_per_col(np.array(measures))
        txt_out = Get_med_25_75_excel(np.array(measures))
        
        cluster_txt_out = cluster_txt_out + txt_out
        # cluster_txt_out.append(txt_out)
        
    tmp_out_list = cluster_name + cluster_txt_out
    output_list.append(tmp_out_list)
    
# print(header_list)
# print(output_list)


    
if ('Summary_Tables' not in os.listdir(my_path)):
    os.mkdir('Summary_Tables')
    
csv_path = os.path.join(os.getcwd(), 'Summary_Tables', 'Offcluster_subgroup_concordance.csv')
with open(csv_path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(output_list)         
    
