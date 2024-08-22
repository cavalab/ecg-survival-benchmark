# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:58:38 2024

@author: CH242985
"""

print('THIS DOES NOT WORK FOR TEST SET ~= TRAIN SET!')
print('TODO: Check for Code15')
print('TODO: ')

import os
import numpy as np
import json
import h5py
import pandas as pd
pd.Series.is_monotonic = pd.Series.is_monotonic_increasing

from pycox.evaluation import EvalSurv
import csv

from tqdm import tqdm

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
                # print(os.listdir(Eval_Data_Path))
                # okay, we made it to the evaluation set.
                
                # store name and data folder
                Model_Dict = {}
                Model_Dict['Name'] = Model_Name                                 
                Model_Dict['Type'] = Model_Name .split('_')[0]    
                Model_Dict['Eval_Data_Folder'] = Eval_Data_Folder               
                
                
                json_path = os.path.join(Eval_Data_Path, 'Eval_Args.txt')   
                with open(json_path) as f:
                    tmp = json.load(f)
                # print(tmp)
                
                # store pycox model
                if ('pycox_mdl' not in tmp.keys()): # PyCox_mdl
                    Model_Dict['pycox_mdl'] = 'None'                               
                else:
                    if (tmp['pycox_mdl'] == 'CoxPH'):
                        Model_Dict['pycox_mdl'] = 'DeepSurv'
                    else:
                        Model_Dict['pycox_mdl'] = tmp['pycox_mdl']
                   
                # store horizon
                if ('horizon' not in tmp.keys()):
                    Model_Dict['horizon'] = 'None'
                else:
                    Model_Dict['horizon'] = tmp['horizon']
                
                # store random seed
                Model_Dict['Rand_Seed'] = tmp['Rand_Seed']
                
                # Now we pull the file
                h5py_path = os.path.join(Eval_Data_Path, 'Stored_Model_Output.hdf5')
                with h5py.File(h5py_path, "r") as f:
                    keys = [key for key in f.keys()]
                    print('--\n\n')
                    for i,k in enumerate(keys):
                        print(i,k,f[k][()].shape)
                        
                    surv = f['surv'][()]
                    disc_y_t = f['disc_y_t'][()]
                    disc_y_e = f['disc_y_e'][()]
                    disc_y_e_bool = np.array(disc_y_e,dtype=bool)
                    sample_time_points = f['sample_time_points'][()]
                    
                    
                    
                    # Now we need to identify how to split
                    column_names = np.array(  [k.decode('UTF-8') for k in f['Test_Col_Names'][()]  ])
                    
                    y_test = f['y_test'][()]
                    
                    if ('MIMIC' in Eval_Data_Folder):
                        Ages = y_test[:,np.where(column_names=='Age')].squeeze()
                        Genders = y_test[:,np.where(column_names=='Gender')].squeeze()
                    elif('Code' in Eval_Data_Folder)    :
                        Ages = y_test[:,np.where(column_names=='age')].squeeze()
                        Genders = y_test[:,np.where(column_names=='is_male')].squeeze()
                    
                    Age_Brackets = [[0, 30], [30, 45], [45, 60], [60,999]]
                    
                    for gender in np.unique(Genders):
                        for Age_Bracket in Age_Brackets:
                            
                            
                            cat_1 = np.where(Genders==gender)
                            cat_2 = np.where(Ages>=Age_Bracket[0])
                            cat_3 = np.where(Ages<Age_Bracket[1])
                            
                            inters_1 = np.intersect1d(cat_1,cat_2)
                            inters_2 = np.intersect1d(inters_1, cat_3)
                            
                            tmp_disc_y_t = disc_y_t[inters_2]
                            tmp_disc_y_e_bool = disc_y_e_bool[inters_2]
                            tmp_surv = surv[inters_2]
                            
                            tmp_surv_df = pd.DataFrame(np.transpose(tmp_surv)) 
                            
                            # bookkeeping
                            subgroup_name = str(gender) + ',' + str(Age_Bracket) +' concordance'
                            print(len(inters_2))

                            concordance, brier, chance  = get_surv_briercordance(tmp_disc_y_t, tmp_disc_y_e_bool, tmp_surv_df, [999], sample_time_points)
                            # concordance = concordance[
                            # brier = brier[-1]
                            # chance = chance[-1]

                            Model_Dict[subgroup_name] = concordance[-1]
                            
                            # track subgroup names for later
                            if (subgroup_name not in Subgroups_Name_List):
                                Subgroups_Name_List.append(subgroup_name)
                    
                    
                    
            Model_Dict_List.append(Model_Dict)
                
# %% okay, we have it sorted. kinda.
# Now we cluster models by matching type, test folder, (train is out for now), and subgroup
inds_clustered = [] # track which dicts have been clustered
clusters = [] # list of lists of inds
cluster_names = [] # list of names (which happen to be lists)

for ind, entry in enumerate(Model_Dict_List):
    if (ind in inds_clustered): # skip if already clustered
        continue
    
    cluster_criteria = ['Type', 'pycox_mdl', 'horizon', 'Eval_Data_Folder']
    
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
    # output: median, 25-75th percentile
    med = np.median(Data)
    p25 = np.percentile(Data,25)
    p75 = np.percentile(Data,75)
    
    out_list = [str(np.round(med,decimals=2)), "({:.2f}".format(p25) + ", {:.2f})".format(p75)]
    
    return out_list
    
subgroup_headers = []
for k in Subgroups_Name_List:
    subgroup_headers = subgroup_headers + [k]
    subgroup_headers = subgroup_headers + ['25-75']

header_list = cluster_criteria + subgroup_headers
output_list = [header_list]
for cluster_inds, cluster_name in zip(clusters, cluster_names):
    
    cluster_txt_out = []
    for subgroup in Subgroups_Name_List:
        
        measures = [ Model_Dict_List[ind][subgroup] for ind in cluster_inds ]
        txt_out = Get_med_25_75_percentile_per_col(np.array(measures))
        
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
    
