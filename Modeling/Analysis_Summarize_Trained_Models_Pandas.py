# -*- coding: utf-8 -*-
"""
The goal is to summarize all the 'eval' outputs in 'trained files'

"""

# import collections
# collections.Callable = collections.abc.Callable


import os, csv
import numpy as np
import json
import h5py
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

Trained_Models_Path = os.path.join(os.getcwd(), 'Trained_Models')

Model_Args_List = [] # Per model evaluation, create two dictionaries. One with train parameters, one with evaluation metrics
Model_Measures_List = []
Targs  = [1,2,5,10, 99] # time horizons (yrs) we care about. for each get nearest AUROC, AUPRC, etc.


# Per training Data Source
for Train_Source in os.listdir(Trained_Models_Path):
    
    # Get Train Path or continue
    Train_Path = os.path.join(Trained_Models_Path, Train_Source)
    if (os.path.isdir(Train_Path) == False):
        continue
    
    # Per model trained
    for Model_Name in os.listdir(Train_Path):
        
        # Get model path or continue
        Model_Path = os.path.join(Train_Path, Model_Name)
        if (os.path.isdir(Model_Path) == False):
            continue
        
        # Okay, so there is a folder that should have some args in it
        # prep default dictionaries for this model. Worst case they'll be empty.
        model_param_dict = {}
        model_measure_dict = {}
        
        # 0. Pull train args or continue
        if (Model_Name.startswith('XGB') == True):
            model_param_dict['Model_Architecture'] = 'XGB'
        else:
            arg_path = os.path.join(Model_Path, 'Train_Args.txt')
            if (os.path.isfile(arg_path) == False):
                continue
            else:
                with open(arg_path) as f:
                    tmp = json.load(f)
                    # store the train args
                    for k in tmp.keys():
                        model_param_dict[k] = tmp[k]   
                    model_param_dict['Model_Architecture'] = model_param_dict['Model_Name'].split('_')[0]
        
        # 1. Grab Training_Progress CSV -> Find Epoch when validation stopped improving
        Train_Progress_CSV_Path = os.path.join(Model_Path, 'Training_Progress.csv')
        try:
            data = np.genfromtxt(Train_Progress_CSV_Path, delimiter=',') # should be all float
            best_epoch = int(data[np.argmin(data[:,2]),0])
            model_param_dict['Best_Epoch'] = best_epoch
            
            # how many epochs could we have waited?
            # in other words, per value, how many entries were there between it and the previous min
            tmp = data[:,2]
            res = []
            for i,k in enumerate(tmp):
                if (i==0):
                    tmp_out = 0
                else:
                    tmp_out = data[i,0] - data[np.argmin(tmp[:i]),0]
                res = res + [tmp_out]
            
            model_param_dict['Min_Early_Stop_To_Not_Miss'] = int(max(res[:best_epoch+1]))

        except:
            print('Error with train progress: only 1 epoch or couldnt open : ' + Train_Progress_CSV_Path)
            
            # don't store anything in dictionary - will be auto-populated with 'no entry' later
            pass
        
        
        
        
        # Get Eval Path or continue. Eval stores _all_ model evaluations in subfolders.
        Eval_Path = os.path.join(Model_Path, 'Eval')
        
        # if no eval path, add empty dictionary for metric, save both
        if (os.path.isdir(Eval_Path) == False):
            model_measure_dict = {}
            Model_Args_List.append(model_param_dict)
            Model_Measures_List.append(model_measure_dict)
            continue
            
        # Per (Test_Folder in Eval_Path)
        for Test_Folder in os.listdir(Eval_Path):
            Test_Path = os.path.join(Eval_Path, Test_Folder)
            
            if (os.path.isdir(Test_Path) == False): 
                continue
            
            # We have at least one evaluation. Copy the model training params and overwrite them with eval params
            eval_param_dict = model_param_dict.copy()
            arg_path = os.path.join(Test_Path, 'Eval_Args.txt')
            if (os.path.isfile(arg_path) == False):
                continue
            else:
                with open(arg_path) as f:
                    tmp = json.load(f)
                    for k in tmp.keys():
                        eval_param_dict[k] = tmp[k]    
        
            # prep measures dictionary
            model_measure_dict = {}
            Measures_Path = os.path.join(Test_Path, 'Stored_Model_Output.hdf5')
            with h5py.File(Measures_Path, "r") as f:
                print(Test_Path)
                # keys = [key for key in f.keys()]
                # for i,k in enumerate(keys):
                #     print(i,k,f[k][()].shape)
                
                
                AUROC = f['AUROC'][()]
                AUPRC = f['AUPRC'][()]
                BS_Brier = f['bootstrap_briers'][()].transpose() # bootstrapped
                BS_Conc  = f['bootstrap_concordances'][()].transpose()
                RS_Brier = f['ipcw_brier_store_all_ecg'][()]
                RS_Conc = f['concordance_store_all_ecg'][()] # right-censored
                Times = f['sample_time_points'][()]
            
            # okay. now for every time point we care about we 1) find the nearest stored entry, 2) log the corresponding time, 3) store all the measures
            
            # per time point, get the nearest indices
            Nearest_Inds = []
            for Time_Point in Targs:
                nearest_ind =  np.argmin(abs(Times - Time_Point))
                
                # if we can find a time point within 1 year of the target, grab data.
                # if ( abs( Times[nearest_ind] - Time_Point) < 1):
                #     Nearest_Inds.append(nearest_ind)
                    
                model_measure_dict['T' + str(Time_Point) + ' ' + 'AUROC']    = AUROC[nearest_ind]
                model_measure_dict['T' + str(Time_Point) + ' ' + 'AUPRC']    = AUPRC[nearest_ind]
                model_measure_dict['T' + str(Time_Point) + ' ' + 'RS_Brier'] = RS_Brier[nearest_ind]
                model_measure_dict['T' + str(Time_Point) + ' ' + 'RS_Conc']  = RS_Conc[nearest_ind]
                
                for i,k in enumerate(BS_Brier[nearest_ind]):
                    model_measure_dict['T' + str(Time_Point) + ' ' + 'BS_Brier' + ' ' + str(i)] = k
                    
                for i,k in enumerate(BS_Conc[nearest_ind]):
                    model_measure_dict['T' + str(Time_Point) + ' ' + 'BS_Conc' + ' ' + str(i)] = k
                    

                # else:
                #     print('Nearest time point to ' + str(Time_Point) + ' was ' + str(Times[nearest_ind]) + '; leaving blank')
            
            # now store the arglist and the measures to the original lists
            Model_Args_List.append(eval_param_dict)
            Model_Measures_List.append(model_measure_dict)
            
# Okay, now that we've processed all the files, turn that into something to save out to a csv
# both Model_Args_List and Model_Measures_List contain dictionaries. They might not have the same entries, so we should populate missing entries with 'no entry'

if (len(Model_Args_List) != len(Model_Measures_List)):
    print('dict mismatch error!')
    breakpoint()
    
# 1. Consolidate entries, even if they're blank
for temp_list in [Model_Args_List, Model_Measures_List]:
    
    for temp_dict in temp_list:
        for temp_key in temp_dict.keys():
            
            for temp_dict2 in temp_list:
                if temp_key not in temp_dict2.keys():
                    
                    temp_dict2[temp_key] = 'No Entry'
                    
# Even though each dictionary has the same keys, they aren't in the same order.
# Get headers = Keys
Header_Model_Arg_List = [k for k in Model_Args_List[0].keys()]
Header_Model_Mes_List = [k for k in Model_Measures_List[0].keys()]
      
# save the output as a list
out_list = []
out_list.append(Header_Model_Arg_List + Header_Model_Mes_List)

# parse both args and measures lists. They should be 1 to 1.
# for each pair of dicts pulled, use the headers to query them so you get all the outputs in the same order
# ... and save everything as a string
for m,n in zip(Model_Args_List, Model_Measures_List): 
    
    temp_list = [str(m[k]) for k in Header_Model_Arg_List] + [str(n[k]) for k in Header_Model_Mes_List]# line up keys here

    out_list.append(temp_list)

# Save to a folder
tables_path = os.path.join(os.getcwd(), 'Summary_Tables')
if (os.path.isdir(tables_path) == False):
    os.mkdir(tables_path)
      
Out_File = os.path.join(tables_path, 'Trained_Model_Summary.csv')
# np.savetxt(Out_File, temp_list, delimiter=",")
# https://stackoverflow.com/questions/2084069/create-a-csv-file-with-values-from-a-python-list         
with open(Out_File, 'w',newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(out_list)
    

# %% Okay, we've summarized the data. 
# Now that we have everything in a dictionary, we may as well run the processing here instead of exporting-then-re-importing everything
# Prep some useful functions

def Get_Data_RS(Model_Measures_List, model_inds, query_list):
    # Get Data across random seeds
    # input:  list of all model measure dictionaries, which indices of that we care about, which queries we want within each dict
    # output: m x q array of measure values. m - model, q - query.
    tmp_data_outer = []
    for b in model_inds:
        tmp = Model_Measures_List[b] # model_measures_list is a list of dicts, each mapping queries to values
        tmp_data_inner = []    
        for q in query_list:
            tmp_data_inner.append(tmp[q])
        tmp_data_outer.append(tmp_data_inner)
    tmp_data_outer = [k for k in tmp_data_outer if 'No Entry' not in k] # sometimes a model fails and has no measures -> trash
    tmp_data = np.array(tmp_data_outer)
    
    return tmp_data

def Get_med_25_75_percentile_per_col(Data):
    # input: m x n array
    # output: string. median_col_0, (25-75), median_col_1, (25-75), etc.
    out_list = []
    try:
        for col in range(tmp_data.shape[1]):
            tmp_col = tmp_data[:,col]
            med = np.median(tmp_col)
            p25 = np.percentile(tmp_col,25)
            p75 = np.percentile(tmp_col,75)
            out_list.append(np.round(med,decimals=2))
            app_str = "({:.2f}".format(p25) + ", {:.2f})".format(p75)
            out_list.append(app_str)
    except:
        out_list = [' ',' ']
        pass
    return out_list

def Get_Data_BS(Model_Measures_List, model_inds, query_list):
    # Get Data across bootstraps and random seeds
    # input:  list of all model measure dictionaries, which indices of that we care about, which queries we want within each dict
    # output: m x q array of measure values. m - models * bootstraps, q - query.

    # each entry of query_list is a list of dictionary keys for one time point. with each key indexing a bootstrap.
    # per query_list, parse through all models and add an entry per bootstrap. then transpose.
    tmp_data_outer = []
    for q in query_list:
        tmp_data_inner = []
        for m in model_inds:
            tmp = Model_Measures_List[m]
            for bs in q:
                tmp_data_inner.append(tmp[bs])
        tmp_data_outer.append(tmp_data_inner)
    tmp_data = np.array(tmp_data_outer)
    tmp_data = np.transpose(tmp_data)
    
    # now 100 x 5. parse those 100, if 'no entry' delete, and convert all remaining to float
    # tmp_data_outer = [k for k in tmp_data_outer if 'No Entry' not in k]

    if (tmp_data.dtype != np.float64):
        ind_to_keep = []
        for i,k in enumerate(tmp_data):
            if ('No Entry' not in k):
                ind_to_keep.append(i)
        tmp_data = tmp_data[ind_to_keep,:] 
        tmp_data = tmp_data.astype(np.float64)

            

    return tmp_data

# %% Now, make some .csv tables where (train_set == test_set)

# We want to make 3 tables
# 1: AUROC median and 25-75th percentile over the random seeds per 
# 3: Concordance + Brier across random seeds (RS)
# 4: Concordance + Brier across random seeds AND bootstraps (BS)

# First, prep table headers and find which keys we care about
All_header = ['Train_Data','Test_Data','Architecture','Task'] 

AUROC_header = ['T1 AUROC', '(25-75th%)', 'T2 AUROC', '(25-75th%)', 'T5 AUROC', '(25-75th%)', 'T10 AUROC', '(25-75th%)']
AUROC_query = ['T1 AUROC', 'T2 AUROC', 'T5 AUROC', 'T10 AUROC']
AUROC_out_rows = []

AUPRC_header = ['T1 AUPRC', '(25-75th%)', 'T2 AUPRC', '(25-75th%)', 'T5 AUPRC', '(25-75th%)', 'T10 AUPRC', '(25-75th%)']
AUPRC_query = ['T1 AUPRC', 'T2 AUPRC', 'T5 AUPRC', 'T10 AUPRC']
AUPRC_out_rows = []

# brier/concordance over random seeds (per-ecg)
ConcordanceRS_header = ['T1 Concordance', '(25-75th%)', 'T2 Concordance', '(25-75th%)', 'T5 Concordance', '(25-75th%)', 'T10 Concordance', '(25-75th%)', 'Concordance', '(25-75th%)']
ConcordanceRS_query = ['T1 RS_Conc', 'T2 RS_Conc', 'T5 RS_Conc', 'T10 RS_Conc', 'T99 RS_Conc']
ConcordanceRS_out_rows = []

BrierRS_header = ['T1 Brier', '(25-75th%)', 'T2 Brier', '(25-75th%)', 'T5 Brier', '(25-75th%)', 'T10 Brier', '(25-75th%)', 'Brier', '(25-75th%)']
BrierRS_query = ['T1 RS_Brier', 'T2 RS_Brier', 'T5 RS_Brier', 'T10 RS_Brier', 'T99 RS_Brier']
BrierRS_out_rows = []

# bootstraps (average over RS and bootstrap)
ConcordanceBS_header = ['T1 concordance', '(25-75th%)', 'T2 concordance', '(25-75th%)', 'T5 concordance', '(25-75th%)', 'T10 concordance', '(25-75th%)', 'All-Time concordance', '(25-75th%)']
ConcordanceBS_queries = [] # we now have lists per time point
for T in [1,2,5,10,99]:
    tmp = []
    for i,k in enumerate(model_measure_dict.keys()):
        if ('T'+str(T)+' ') in k:
            if 'BS_Conc' in k:
                tmp.append(k)
    ConcordanceBS_queries.append(tmp)
ConcordanceBS_out_rows = []

BrierBS_header = ['T1 brier', '(25-75th%)', 'T2 brier', '(25-75th%)', 'T5 brier', '(25-75th%)', 'T10 brier', '(25-75th%)', 'All-Time brier', '(25-75th%)']
BrierBS_queries = [] # we now have lists per time point
for T in [1,2,5,10,99]:
    tmp = []
    for i,k in enumerate(model_measure_dict.keys()):
        if ('T'+str(T)+' ') in k:
            if 'BS_Brier' in k:
                tmp.append(k)
    BrierBS_queries.append(tmp)
BrierBS_out_rows = []


# %% Clustering trained models
# give me metrics as med +/- percentile for a set of models
# where the set is one where the following parameters all match
cluster_keys = ['Train_Folder', 'Test_Folder', 'horizon', 'pycox_mdl', 'Model_Architecture', 'val_covariate_col_list']


# Then, cluster models and summarize metrics per cluster
model_cluster_names = []     
processed_model_inds = []  # track which models we've clustered
for i,k in enumerate(Model_Args_List):  # parse the args k per model evalution
    # if (k['Train_Folder'] != k['Test_Folder']): # exclude model evaluations where Test and Train aren't from the same dataset
    #     continue
    if i in processed_model_inds: # exclude models we've already clustered
        continue
    model_cluster_inds = [i]
    for j, m in enumerate(Model_Args_List): # 
        if i == j:                          # parse other model evaluations
            continue
        
        # breakpoint()
        for key in cluster_keys: # cluster by
            model_matches = True
            if m[key] != k[key]: # exclude model evaluations that don't match 'k' on train_folder, test_folder, horizon, pycox_model, and model_name [model type]
                model_matches = False
                break
        if (model_matches):
            model_cluster_inds.append(j)
            
    for m in model_cluster_inds:
        processed_model_inds.append(m) # ensure we don't repeat this clustering later
        
    # figure out how to name this cluster
    train_folder = k['Train_Folder']
    test_folder = k['Test_Folder']
    architecture = k['Model_Architecture']
    survmodel = ''
    if k['horizon'] == 'No Entry':
        if (k['pycox_mdl'] == 'CoxPH'):
            survmodel = 'DeepSurv'
        else:
            survmodel = k['pycox_mdl']
    else:
        survmodel = 'Cla-'+str(int(float((k['horizon']))))
        
    # covariates
    tmp = len(k['val_covariate_col_list'].split(','))
    if tmp==2: # demographics
        cov = 'demographic'
    elif tmp > 2:
        cov = 'demographic + mm'
    else:
        cov = 'none'
        
    # number successful models
    tmp_data = Get_Data_RS(Model_Measures_List, model_cluster_inds, AUROC_query) # returns a [models] x [queries] array

    N_Run = len(model_cluster_inds)
    # print(tmp_data)
    # print(tmp_data.shape[0])
    N_Success = tmp_data.shape[0]

    if (N_Success ==0):
        breakpoint()
    
    # store the names
    model_cluster_names.append([train_folder, test_folder, architecture, survmodel, cov, N_Run, N_Success])
    model_name_header = ['train_folder', 'test_folder', 'architecture', 'survmodel', 'cov', 'N_Run', 'N_Success']
            
    # Now get the measures
    tmp_data = Get_Data_RS(Model_Measures_List, model_cluster_inds, AUROC_query) # returns a [models] x [queries] array
    tmp_text = Get_med_25_75_percentile_per_col(tmp_data)
    AUROC_out_rows.append(tmp_text)
    
    tmp_data = Get_Data_RS(Model_Measures_List, model_cluster_inds, AUPRC_query) # returns a [models] x [queries] array
    tmp_text = Get_med_25_75_percentile_per_col(tmp_data)
    AUPRC_out_rows.append(tmp_text)
    
    tmp_data = Get_Data_RS(Model_Measures_List, model_cluster_inds, ConcordanceRS_query) # returns a [models] x [queries] array
    tmp_text = Get_med_25_75_percentile_per_col(tmp_data)
    ConcordanceRS_out_rows.append(tmp_text)
    
    tmp_data = Get_Data_RS(Model_Measures_List, model_cluster_inds, BrierRS_query) # returns a [models] x [queries] array
    tmp_text = Get_med_25_75_percentile_per_col(tmp_data)
    BrierRS_out_rows.append(tmp_text)
    
    tmp_data = Get_Data_BS(Model_Measures_List, model_cluster_inds, ConcordanceBS_queries) # returns a [models x bootstraps] x [queries] array
    tmp_text = Get_med_25_75_percentile_per_col(tmp_data)
    ConcordanceBS_out_rows.append(tmp_text)
    
    tmp_data = Get_Data_BS(Model_Measures_List, model_cluster_inds, BrierBS_queries) # returns a [models x bootstraps] x [queries] array
    tmp_text = Get_med_25_75_percentile_per_col(tmp_data)
    BrierBS_out_rows.append(tmp_text)
    

# Lastly, save out tables

# Main Results tabel - train / task / architecture, AUROC 1-10, AUPRC 1-10, T99 concordance, T99 Brier
header = model_name_header + AUROC_header + AUPRC_header +[ ConcordanceRS_header[-2]] + [ConcordanceRS_header[-1]] +[ BrierRS_header[-2]] + [BrierRS_header[-1]]
out_list = [header]
for row_num in range(len(model_cluster_names)):
    tmp_list = [str(k) for k in model_cluster_names[row_num]] + [str(k) for k in AUROC_out_rows[row_num]] + [str(k) for k in AUPRC_out_rows[row_num]] + [str(ConcordanceRS_out_rows[row_num][-2])] + [str(ConcordanceRS_out_rows[row_num][-1])] + [str(BrierRS_out_rows[row_num][-2])] + [str(BrierRS_out_rows[row_num][-1])]
    out_list.append(tmp_list)
Out_File_Path = os.path.join(tables_path, 'Trained_Model_Main_Result_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)
    
# BrierCordance RS Table - train / task / architecture, concordance RS 1-10,99, Brier RS 1-10, 99
header = model_name_header + ConcordanceRS_header + BrierRS_header
out_list = [header]
for row_num in range(len(model_cluster_names)):
    tmp_list = [str(k) for k in model_cluster_names[row_num]] + [str(k) for k in ConcordanceRS_out_rows[row_num]] + [str(k) for k in BrierRS_out_rows[row_num]] 
    out_list.append(tmp_list)
Out_File_Path = os.path.join(tables_path, 'Trained_Model_BrierConcordance_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)   
    
# BrierCordance BS Table - train / task / architecture, concordance BS 1-10,99, Brier BS 1-10, 99
header = model_name_header + ConcordanceBS_header + BrierBS_header
out_list = [header]
for row_num in range(len(model_cluster_names)):
    tmp_list = [str(k) for k in model_cluster_names[row_num]] + [str(k) for k in ConcordanceBS_out_rows[row_num]] + [str(k) for k in BrierBS_out_rows[row_num]] 
    out_list.append(tmp_list)
Out_File_Path = os.path.join(tables_path, 'Trained_Model_BS_BrierConcordance_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)   
    
# # %% Cross-evaluation. ...
# # ... actually, it looks like we just pull entries from the summary table 
# csv_path = os.path.join(os.getcwd(), 'Summary_Tables', 'Trained_Model_Summary.csv')
# my_data = np.genfromtxt(csv_path, delimiter=',', dtype=str)

# Train_fold_Ind = np.where(my_data[0] == 'Train_Folder')[0][0]
# Test_fold_Ind = np.where(my_data[0] == 'Test_Folder')[0][0]
# horizon_ind = np.where(my_data[0] == 'horizon')[0][0]
# RS_ind = np.where(my_data[0] == 'Rand_Seed')[0][0]
# PyCx_ind = np.where(my_data[0] == 'pycox_mdl')[0][0]

# out_list = [my_data[0]]
# for k in my_data[1:]:
#     if k[Train_fold_Ind] != k[Test_fold_Ind]: # find cases where train and test are different
#         out_list.append(k) # copy them into a new table
        
#         # then find the matching case where they are the same
#         for k2 in my_data[1:]:
#             if k2[Train_fold_Ind] == k2[Test_fold_Ind]: # train and test must match here
#                 match_found = True
#                 for ind in [horizon_ind,RS_ind,PyCx_ind]:
#                     if (k[ind] != k2[ind]):
#                         match_found = False
#                         break
#                 if (match_found):
#                     out_list.append(k2) # copy that case into the table too
            
        
# Out_File_Path = os.path.join(tables_path, 'Trained_Model_Crossval_Table.csv')    
# with open(Out_File_Path, 'w',newline='') as f:
#     wr = csv.writer(f)
#     wr.writerows(out_list)   
    
# %% Stats on things like [does surv differ from class?]
breakpoint()

# does concordance differ for Code15 Surv vs Class?

Model_Args_List[0].keys() # shows list of key options
np.unique(np.array([Model_Args_List[k]['Model_Architecture'] for k in range(len(Model_Args_List))])) # shows unique entry options

def sort_model_inds (inds):
# func - sort models by survival case, architecture, then random seed, in that order (low to high)
    ind_scores = []
    for k in inds:
        ind_score = 0
        if (Model_Args_List[k]['pycox_mdl'] == 'CoxPH'):
            ind_score += 100
        if (Model_Args_List[k]['pycox_mdl'] == 'DeepHit'):
            ind_score += 200
        if (Model_Args_List[k]['pycox_mdl'] == 'LH'):
            ind_score += 300
        if (Model_Args_List[k]['pycox_mdl'] == 'MTLR'):
            ind_score += 400
        if (Model_Args_List[k]['pycox_mdl'] == 'No Entry'):
            if (Model_Args_List[k]['horizon'] == '1.0'):    
                ind_score += 500
            if (Model_Args_List[k]['horizon'] == '2.0'):    
                ind_score += 600
            if (Model_Args_List[k]['horizon'] == '5.0'):    
                ind_score += 700
            if (Model_Args_List[k]['horizon'] == '10.0'):    
                ind_score += 800
                
        if (Model_Args_List[k]['Model_Architecture'] == 'InceptionTime'):
            ind_score += 10
        if (Model_Args_List[k]['Model_Architecture'] == 'Ribeiro'):
            ind_score += 20
        
        ind_score += Model_Args_List[k]['Rand_Seed'] - 10 # rand seed is 10-14
        ind_scores.append(ind_score)
        
    new_order = np.argsort(np.array(ind_scores))
    return [inds[k] for k in new_order]
            
# 1. load data matching specific cases (sort by survival case and then random seed)
def Get_Data_RS_bulk(Measure='', RunFolder='', Cov_List = '', Surv = '', Model_Types = '',debug = False):
    # returns T99 RS_Conc for models matching the above criteria
    # input: Model_Args_List
    # input: Model_Measures_List
    # input: RunFolder - which folder the model must have been trained and tested on
    # input: Cov_List - 'Code15' or 'MIMIC' or 'None'
    # input: Surv - 'Surv', 'Class', or 'Any'
    # input: Model_Types - 'Inception', 'Resnet', 'ECG' or 'Demographic'
    
    # which covariate list must the models match?
    if Cov_List == 'MIMICIV':
        Cov_Criteria = '[1,2]'
    elif Cov_List == 'Code15':
        Cov_Criteria = '[2,5]'
    elif Cov_List == 'None':
        Cov_Criteria = 'No Entry'
        
    if Surv == 'PyCox':
        Surv_Criteria = ['CoxPH', 'DeepHit', 'LH', 'MTLR']
    elif Surv == 'SurvClass':
        Surv_Criteria = ['No Entry']
    elif Surv == 'Any':
        Surv_Criteria = ['CoxPH', 'DeepHit', 'LH', 'MTLR', 'No Entry']
        
    if Model_Types == 'ECG':
        Model_Criteria = ['InceptionTime', 'Ribeiro']
    elif Model_Types == 'Demographic':
        Model_Criteria = ['XGB','ZeroNet']
    elif Model_Types == 'InceptionTime':
        Model_Criteria = ['InceptionTime']
    elif Model_Types == 'Ribeiro':
        Model_Criteria = ['Ribeiro']
        
    # ECG Demo Code15 Surv   
    inds = []
    for i,m in enumerate(Model_Args_List):
        if (m['Train_Folder'] != RunFolder):
            continue
        if (m['Test_Folder'] != RunFolder):
            continue
        if (m['val_covariate_col_list'] != Cov_Criteria): # [2,5] for Code15, [1,2] for MIMIC
            continue
        if (m['Model_Type'] not in Model_Criteria): # inceptionTime and Ribeiro ok
            continue
        if (m['pycox_mdl'] not in  Surv_Criteria): # surv models only
            continue
        inds.append(i)
    inds = sort_model_inds(inds)
        
    if (debug):
        for i,k in enumerate(inds):
            print(Model_Args_List[k])
            
    return Get_Data_RS(Model_Measures_List,inds,[Measure])

def quick_median_IQR(arr):
    tmp1 = np.mean(arr)
    tmp2 = np.percentile(arr, 25)
    tmp3 = np.percentile(arr, 75)
    print(tmp1, tmp2, tmp3)
    # return (tmp1, tmp2, tmp3)

# Code15 

# Surv vs Class - can't do paired comparisons
Measure = 'T99 RS_Conc'
a = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='SurvClass', Model_Types='ECG', debug = True)
b = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='None', Surv='SurvClass', Model_Types='ECG')
c = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='SurvClass', Model_Types='Demographic')
d = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='PyCox', Model_Types='ECG')
e = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='None', Surv='PyCox', Model_Types='ECG')
f = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='PyCox', Model_Types='Demographic')

from scipy.stats import mannwhitneyu
mannwhitneyu(a,d)
mannwhitneyu(b,e)
mannwhitneyu(c,f)

quick_median_IQR(e)

from scipy.stats import pearsonr
y = [1]*10+[2]*10+[5]*10+[10]*10
pearsonr(a.squeeze(),np.array(y)) # survclass with demographics vs horizon
pearsonr(b.squeeze(),np.array(y)) # survclass vs horizon


# Ribeiro vs InceptionTime - CAN do paired comparisons ... if I sort everything somehow
a = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='Any', Model_Types='InceptionTime', debug = True)
b = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='None', Surv='Any', Model_Types='InceptionTime')
c = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='Any', Model_Types='Demographic')
d = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='Any', Model_Types='Ribeiro')
e = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='None', Surv='Any', Model_Types='Ribeiro')
f = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'Code15', Cov_List='Code15', Surv='Any', Model_Types='Demographic')
    
from scipy.stats import wilcoxon
wilcoxon (a,d) # ecg + demo
wilcoxon (b,e) # ecg 



# MIMIC
a = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='SurvClass', Model_Types='ECG')
b = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='None', Surv='SurvClass', Model_Types='ECG')
c = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='SurvClass', Model_Types='Demographic')
d = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='PyCox', Model_Types='ECG')
e = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='None', Surv='PyCox', Model_Types='ECG')
f = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='PyCox', Model_Types='Demographic')

mannwhitneyu(a,d)
mannwhitneyu(b,e)
mannwhitneyu(c,f)



# Ribeiro vs InceptionTime - CAN do paired comparisons ... if I sort everything somehow
a = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='Any', Model_Types='InceptionTime')
b = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='None', Surv='Any', Model_Types='InceptionTime')
c = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='Any', Model_Types='Demographic', debug = True)
d = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='Any', Model_Types='Ribeiro')
e = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='None', Surv='Any', Model_Types='Ribeiro')
f = Get_Data_RS_bulk(Measure=Measure, RunFolder = 'MIMICIV', Cov_List='MIMICIV', Surv='Any', Model_Types='Demographic')
  


wilcoxon (a,d) # ecg + demo
wilcoxon (b,e) # ecg       
pearsonr(a.squeeze(),np.array(y)) # survclass with demographics vs horizon
pearsonr(b.squeeze(),np.array(y)) # survclass vs horizon

# %%  Now convert everything over to a pandas dataframe
dataframe = pd.DataFrame(out_list[1:], columns=Header_Model_Arg_List+Header_Model_Mes_List)

# exploring the dataframe
for i,k in enumerate(dataframe.columns):
    print (i,k)
    
dataframe['pycox_mdl']

pycox_mdls = dataframe['pycox_mdl'].unique()
Test_Folders = dataframe['Test_Folder'].unique()
Train_Folders = dataframe['Train_Folder'].unique()
Rand_Seeds = dataframe['Rand_Seed'].unique()
Model_Architectures = dataframe['Model_Architecture'].unique()
Datasets = dataframe['Train_Folder'].unique()
cov_sets = dataframe['val_covariate_col_list'].unique()

#adding to dataframe - survival type
Surv_Model_Types = []
for i,j in zip (dataframe['pycox_mdl'], dataframe['horizon']):
    if (i == 'No Entry'):
        if (float(j) < 10):
            tmp = 'Cla-0'+j[:-2]
        else:
            tmp = 'Cla-'+j[:-2]
    elif (i == 'CoxPH'):
        tmp = 'DeepSurv'
    else:
        tmp = i
    Surv_Model_Types.append(tmp)
surv_model_df = pd.DataFrame(data=Surv_Model_Types, columns=['Surv_Model'])    

dataframe = dataframe.join(surv_model_df)       
dataframe['Concordance'] = dataframe['T99 RS_Conc'].astype(float) 
Surv_Model_Types_un = dataframe['Surv_Model'].unique()
    
# adding to dataframe - survival type + covariate info
Surv_Model_Types_Combo = []
for i,j,k in zip (dataframe['pycox_mdl'], dataframe['horizon'], dataframe['val_covariate_col_list']):
    tmp = ''
    if (len(k)==5):
        tmp = '+ '
    if (i == 'No Entry'):
        if (float(j) < 10):
            tmp += 'Cla-0'+j[:-2]
        else:
            tmp += 'Cla-'+j[:-2]
    elif (i == 'CoxPH'):
        tmp += 'DeepSurv'
    else:
        tmp += i
    Surv_Model_Types_Combo.append(tmp)
tmp_df = pd.DataFrame(data=Surv_Model_Types_Combo, columns=['Cov_Surv_Model'])    
    
dataframe = dataframe.join(tmp_df) 

Architectures = []
for i in dataframe['Model_Architecture']:
    if i == 'Ribeiro':
        Architectures.append('ResNet')
    elif i == 'ZeroNet':
        Architectures.append('Feedforward Net')
    else:
        Architectures.append(i)
dataframe['Architecture'] = Architectures

# Start plotting 

# Code15 with and without covariates
tmp = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])
plt.figure()
ax = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='Concordance', hue = 'Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'])
ax.tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax.set_title('Code-15')
ax.axhline(0.7905, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax.axvline(3.5,ls='--',color='k')
ax.axvline(7.5,ls='--',color='k')
ax.axvline(11.5,ls='--',color='k')
ax.text(12, 0.785, "Best non-ECG", color='r')
ax.set(xlabel=None)
ax.legend(loc='lower right')     

# MIMICIV with and without covariates
tmp = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])
plt.figure()
ax = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='Concordance', hue = 'Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'])
ax.tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax.set_title('MIMIC-IV')
# ax.axhline(0.7905, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
# ax.text(-0.4, 0.786, "Best non-ECG", color='r')
ax.axvline(3.5,ls='--',color='k')
ax.axvline(7.5,ls='--',color='k')
ax.axvline(11.5,ls='--',color='k')
ax.set(xlabel=None)
ax.legend(loc='lower right')  

# Code15 no covariates
tmp = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])
plt.figure()
ax = seaborn.boxplot(data=tmp,x='Surv_Model',y='Concordance', hue = 'Architecture', fill = False)
ax.tick_params(axis='x', rotation=30) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax.set_title('Code15 No Covariates')
ax.axhline(0.7905, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax.text(-0.4, 0.786, "Best non-ECG", color='r')
ax.set(xlabel=None)
ax.legend(loc='lower right')

# Code15 with covariates
tmp = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime","ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])
plt.figure()
ax = seaborn.boxplot(data=tmp,x='Surv_Model',y='Concordance', hue = 'Architecture', fill = False)
ax.tick_params(axis='x', rotation=30) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax.set_title('Code15 with Age/Sex')
ax.set(xlabel=None)
ax.legend(loc='lower right')


# MIMIC no covariates
tmp = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])
plt.figure()
ax = seaborn.boxplot(data=tmp,x='Surv_Model',y='Concordance', hue = 'Architecture', fill = False)
ax.tick_params(axis='x', rotation=30) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax.set_title('MIMICIV No Covariates')
ax.set(xlabel=None)
ax.legend(loc='lower right')

#LOOK AT CATPLOT

# MIMICIV with covariates
tmp = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           * (dataframe["val_covariate_col_list"].isin(["[1,2]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime","ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])
plt.figure()
ax = seaborn.boxplot(data=tmp,x='Surv_Model',y='Concordance', hue = 'Architecture', fill = False)
ax.tick_params(axis='x', rotation=30) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax.set_title('MIMICIV with Age/Sex')
ax.set(xlabel=None)
ax.legend(loc='lower right')


# now let's compare ECG vs ECG + demographics

# MIMIC
tmp1 = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           * (dataframe["val_covariate_col_list"].isin(["[1,2]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime","ResNet"]))
           ].sort_values(['Surv_Model','Architecture', 'Rand_Seed'])['Concordance']

tmp2 = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[1,2]"]))
            * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime","ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])['Concordance']

wilcoxon (tmp1,tmp2) # ecg vs ecg + demo
quick_median_IQR(tmp1)
quick_median_IQR(tmp2)

# Code15
tmp1 = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime","ResNet"]))
           ].sort_values(['Surv_Model','Architecture', 'Rand_Seed'])['Concordance']

tmp2 = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[1,2]"]))
            * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["InceptionTime","ResNet"]))
           ].sort_values(['Surv_Model','Architecture'])['Concordance']

wilcoxon (tmp1,tmp2) # ecg vs ecg + demo
quick_median_95CI(tmp1)
quick_median_95CI(tmp2)

# %%
# non-ecg baselines Code15
tmp = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
            * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["XGB"]))
           ].sort_values(['Surv_Model','Architecture'])
medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
print('xgb code-15')
print(medians)

tmp = dataframe[ (dataframe["Test_Folder"]== "Code15")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
            * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["Feedforward Net"]))
           ].sort_values(['Surv_Model','Architecture'])
medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
print('FF code-15')
print(medians)


# non-ecg baselines MIMIC
tmp = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
            * (dataframe["val_covariate_col_list"].isin(["[1,2]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["XGB"]))
           ].sort_values(['Surv_Model','Architecture'])
medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
print('xgb MIMIC')
print(medians)


tmp = dataframe[ (dataframe["Test_Folder"]== "MIMICIV")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
            * (dataframe["val_covariate_col_list"].isin(["[1,2]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["Feedforward Net"]))
           ].sort_values(['Surv_Model','Architecture'])
medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
print('FF MIMIC')
print(medians)

# non-ecg baselines BCH
tmp = dataframe[ (dataframe["Test_Folder"]== "BCH_ECG")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
            * (dataframe["val_covariate_col_list"].isin(["[2,7]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["XGB"]))
           ].sort_values(['Surv_Model','Architecture'])
medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
print('xgb BCH_ECG')
print(medians)


tmp = dataframe[ (dataframe["Test_Folder"]== "BCH_ECG")
           * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
            * (dataframe["val_covariate_col_list"].isin(["[2,7]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (dataframe["Architecture"].isin(["Feedforward Net"]))
           ].sort_values(['Surv_Model','Architecture'])
medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
print('FF BCH_ECG')
print(medians)


# %% Get a bunch of median / IQR values

# Cross-evaluation no dem
output_lines = []

for Test_Folder in ["BCH_ECG", "Code15", "MIMICIV"]:
    for Train_Folder in ["BCH_ECG", "Code15", "MIMICIV"]:
        
        print('\n')
        print(Train_Folder, Test_Folder)
        tmp = dataframe[ (dataframe["Train_Folder"]== Train_Folder)
                   * (dataframe["Test_Folder"].isin([Test_Folder]))
                    # * (dataframe["val_covariate_col_list"].isin(["[1,2,6,8,10,12,14,16,18,20]"]))
                    * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
                    * (dataframe["Architecture"].isin(["InceptionTime"]))
                    * (dataframe["Surv_Model"].isin(["DeepHit"]))
                   ]#.sort_values(['Surv_Model','Architecture'])
        medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
        p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
        p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
        for i,j,k in zip(medians, p25, p75):
            print(i,j,k)
        output_lines.append([Test_Folder, Train_Folder, i,j,k])
        
output_lines = np.array(output_lines)


# Cross-evaluation w dem
output_lines = []

for Test_Folder in ["BCH_ECG", "Code15", "MIMICIV"]:
    for Train_Folder in ["BCH_ECG", "Code15", "MIMICIV"]:
        
        print('\n')
        print(Train_Folder, Test_Folder)
        tmp = dataframe[ (dataframe["Train_Folder"]== Train_Folder)
                   * (dataframe["Test_Folder"].isin([Test_Folder]))
                    * (dataframe["val_covariate_col_list"].isin(["[1,2]", "[2,5]", "[2,7]"]))
                    # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
                    * (dataframe["Architecture"].isin(["InceptionTime"]))
                    * (dataframe["Surv_Model"].isin(["DeepHit"]))
                   ]#.sort_values(['Surv_Model','Architecture'])
        medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
        p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
        p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
        for i,j,k in zip(medians, p25, p75):
            print(i,j,k)
        output_lines.append([Test_Folder, Train_Folder, i,j,k])
        
output_lines = np.array(output_lines)



# Cross-evaluation w best non-ECG dem
output_lines = []

for Test_Folder in ["BCH_ECG", "Code15", "MIMICIV"]:
    for Train_Folder in ["BCH_ECG", "Code15", "MIMICIV"]:
        
        if Train_Folder == 'BCH_ECG':
            targ_surv_model = 'DeepHit'
            targ_arch = 'Feedforward Net'
        else:
            targ_surv_model = 'Cla-02' # MIMIC-IV and Code-15 both did best with FF-CLA2
            targ_arch = 'Feedforward Net'

        
        print('\n')
        print(Train_Folder, Test_Folder)
        tmp = dataframe[ (dataframe["Train_Folder"]== Train_Folder)
                   * (dataframe["Test_Folder"].isin([Test_Folder]))
                    * (dataframe["val_covariate_col_list"].isin(["[1,2]", "[2,5]", "[2,7]"]))
                    # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
                    * (dataframe["Architecture"].isin([targ_arch]))
                    * (dataframe["Surv_Model"].isin([targ_surv_model]))
                   ]#.sort_values(['Surv_Model','Architecture'])
        medians =  tmp.groupby(['Surv_Model'])['Concordance'].median()
        p25 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.25)
        p75 =  tmp.groupby(['Surv_Model'])['Concordance'].quantile(0.75)
        for i,j,k in zip(medians, p25, p75):
            print(i,j,k)
        output_lines.append([Test_Folder, Train_Folder, i,j,k])
        
output_lines = np.array(output_lines)