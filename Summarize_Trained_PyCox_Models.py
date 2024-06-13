# -*- coding: utf-8 -*-
"""
The goal is to summarize all the 'eval' outputs in 'trained files'

"""

print('Model Summary only works for PyCox models')
# import collections
# collections.Callable = collections.abc.Callable


import os, csv
import numpy as np
import json
import h5py


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
            Measures_Path = os.path.join(Test_Path, 'Surv_Outputs.hdf5')
            with h5py.File(Measures_Path, "r") as f:
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
        tmp = Model_Measures_List[b]
        tmp_data_inner = []    
        for q in query_list:
            tmp_data_inner.append(tmp[q])
        tmp_data_outer.append(tmp_data_inner)
    tmp_data = np.array(tmp_data_outer)
    return tmp_data

def Get_med_25_75_percentile_per_col(Data):
    # input: m x n array
    # output: string. median_col_0, (25-75), median_col_1, (25-75), etc.
    out_list = []
    for col in range(tmp_data.shape[1]):
        tmp_col = tmp_data[:,col]
        med = np.median(tmp_col)
        p25 = np.percentile(tmp_col,25)
        p75 = np.percentile(tmp_col,75)
        out_list.append(np.round(med,decimals=2))
        app_str = "({:.2f}".format(p25) + ", {:.2f})".format(p75)
        out_list.append(app_str)
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

# Then, cluster models and summarize metrics per cluster
model_cluster_names = []     
processed_model_inds = []  # track which models we've clustered
for i,k in enumerate(Model_Args_List):  # parse the args k per model evalution
    if (k['Train_Folder'] != k['Test_Folder']): # exclude model evaluations where Test and Train aren't from the same dataset
        continue
    if i in processed_model_inds: # exclude models we've already clustered
        continue
    model_cluster_inds = [i]
    for j, m in enumerate(Model_Args_List): # 
        if i == j:                          # parse other model evaluations
            continue
        for key in ['Train_Folder', 'Test_Folder', 'horizon', 'pycox_mdl', 'Model_Architecture']:
            model_matches = True
            if m[key] != k[key]: # exclude model evaluations that don't match 'k' on train_folder, test_folder, horizon, pycox_model, and model_name [model type]
                model_matches = False
                break
        if (model_matches):
            model_cluster_inds.append(j)
            
    for m in model_cluster_inds:
        processed_model_inds.append(m) # ensure we don't repeat this clustering later
        
    # figure out how to name this cluster
    tr_d = k['Train_Folder']
    te_d = k['Test_Folder']
    a = k['Model_Architecture']
    if k['horizon'] == 'No Entry':
        if (k['pycox_mdl'] == 'CoxPH'):
            m = 'DeepSurv'
        else:
            m = k['pycox_mdl']
    else:
        m = 'Cla-'+str(int(float((k['horizon']))))
            
    # store the names
    model_cluster_names.append([tr_d,te_d,a,m])
            
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
header = ['d','m','a'] + AUROC_header + AUPRC_header +[ ConcordanceRS_header[-2]] + [ConcordanceRS_header[-1]] +[ BrierRS_header[-2]] + [BrierRS_header[-1]]
out_list = [header]
for row_num in range(len(model_cluster_names)):
    tmp_list = [str(model_cluster_names[row_num][0])] + [str(model_cluster_names[row_num][2])] + [str(model_cluster_names[row_num][3])] + [str(k) for k in AUROC_out_rows[row_num]] + [str(k) for k in AUPRC_out_rows[row_num]] + [str(ConcordanceRS_out_rows[row_num][-2])] + [str(ConcordanceRS_out_rows[row_num][-1])] + [str(BrierRS_out_rows[row_num][-2])] + [str(BrierRS_out_rows[row_num][-1])]
    out_list.append(tmp_list)
Out_File_Path = os.path.join(tables_path, 'Trained_Model_Main_Result_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)
    
# BrierCordance RS Table - train / task / architecture, concordance RS 1-10,99, Brier RS 1-10, 99
header = ['d','m','a'] + ConcordanceRS_header + BrierRS_header
out_list = [header]
for row_num in range(len(model_cluster_names)):
    tmp_list = [str(model_cluster_names[row_num][0])] + [str(model_cluster_names[row_num][2])] + [str(model_cluster_names[row_num][3])] + [str(k) for k in ConcordanceRS_out_rows[row_num]] + [str(k) for k in BrierRS_out_rows[row_num]] 
    out_list.append(tmp_list)
Out_File_Path = os.path.join(tables_path, 'Trained_Model_BrierConcordance_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)   
    
# BrierCordance BS Table - train / task / architecture, concordance BS 1-10,99, Brier BS 1-10, 99
header = ['d','m','a'] + ConcordanceBS_header + BrierBS_header
out_list = [header]
for row_num in range(len(model_cluster_names)):
    tmp_list = [str(model_cluster_names[row_num][0])] + [str(model_cluster_names[row_num][2])] + [str(model_cluster_names[row_num][3])] + [str(k) for k in ConcordanceBS_out_rows[row_num]] + [str(k) for k in BrierBS_out_rows[row_num]] 
    out_list.append(tmp_list)
Out_File_Path = os.path.join(tables_path, 'Trained_Model_BS_BrierConcordance_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)   
    
# %% Cross-evaluation. ...
# ... actually, it looks like we just pull entries from the summary table 
csv_path = os.path.join(os.getcwd(), 'Summary_Tables', 'Trained_Model_Summary.csv')
my_data = np.genfromtxt(csv_path, delimiter=',', dtype=str)

Train_fold_Ind = np.where(my_data[0] == 'Train_Folder')[0][0]
Test_fold_Ind = np.where(my_data[0] == 'Test_Folder')[0][0]
horizon_ind = np.where(my_data[0] == 'horizon')[0][0]
RS_ind = np.where(my_data[0] == 'Rand_Seed')[0][0]
PyCx_ind = np.where(my_data[0] == 'pycox_mdl')[0][0]

out_list = [my_data[0]]
for k in my_data[1:]:
    if k[Train_fold_Ind] != k[Test_fold_Ind]: # find cases where train and test are different
        out_list.append(k) # copy them into a new table
        
        # then find the matching case where they are the same
        for k2 in my_data[1:]:
            if k2[Train_fold_Ind] == k2[Test_fold_Ind]: # train and test must match here
                match_found = True
                for ind in [horizon_ind,RS_ind,PyCx_ind]:
                    if (k[ind] != k2[ind]):
                        match_found = False
                        break
                if (match_found):
                    out_list.append(k2) # copy that case into the table too
            
        
Out_File_Path = os.path.join(tables_path, 'Trained_Model_Crossval_Table.csv')    
with open(Out_File_Path, 'w',newline='') as f:
    wr = csv.writer(f)
    wr.writerows(out_list)   
    
