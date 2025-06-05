# -*- coding: utf-8 -*-
"""
The goal is to summarize all the 'eval' outputs in 'trained files'

"""

# import collections
# collections.Callable = collections.abc.Callable
# %% 1. pull data and create dictionary

import os, csv
import numpy as np
import json
import h5py
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from scipy.stats import wilcoxon
from scipy.stats import pearsonr

# %%

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
            model_param_dict['Best_Val_Err'] = np.min(data[:,2])

        except:
            print('Error with train progress: only 1 epoch or couldnt open : ' + Train_Progress_CSV_Path)
            
            # don't store anything in dictionary - will be auto-populated with 'no entry' later
            pass
        
        
        
        
        # Get Eval Path or continue. Eval stores _all_ model evaluations in subfolders.
        Eval_Path = os.path.join(Model_Path, 'Eval')
        
        # if no eval path or missing model output, add empty dictionary for metrics, then save both dicts
        skip_eval_dir = False
        
        if (os.path.isdir(Eval_Path) == False): # no eval folder
            skip_eval_dir = True
        else:                                   # has eval folder: check for any model outputs
            skip_eval_dir = True
            for root,dirs,files in os.walk(Eval_Path):
                if 'Stored_Model_Output.hdf5' in files: # any model outputs: continue
                    skip_eval_dir = False
                    break
        
        if (skip_eval_dir):
            print('could not eval', Eval_Path)
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
                # BS_Brier = f['bootstrap_briers'][()].transpose() # bootstrapped
                # BS_Conc  = f['bootstrap_concordances'][()].transpose()
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
                
                # for i,k in enumerate(BS_Brier[nearest_ind]):
                #     model_measure_dict['T' + str(Time_Point) + ' ' + 'BS_Brier' + ' ' + str(i)] = k
                # for i,k in enumerate(BS_Conc[nearest_ind]):
                #     model_measure_dict['T' + str(Time_Point) + ' ' + 'BS_Conc' + ' ' + str(i)] = k
                    

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
    
    
# %% Run analysis with pandas
tables_path = os.path.join(os.getcwd(), 'Summary_Tables')
Out_File = os.path.join(tables_path, 'Trained_Model_Summary.csv')

df = pd.read_csv(Out_File) # pull the summary file we just saved

df.loc[ df['Model_Architecture'] == 'Ribeiro', 'Model_Architecture'] = 'ResNet'
df.loc[ df['Model_Architecture'] == 'ZeroNet', 'Model_Architecture'] = 'Feedforward Net'

# add to dataframe - survival type + covariate info
CC_or_DS = [] # mark as classifier-cox or deep survival
Surv_Model_Types_Combo = []
CovTypes = [] # and also mark type of covariate info (none, Age/Sex, Machine)
for i,j,k in zip (df['pycox_mdl'], df['horizon'], df['covariates']):
    tmp = ''
    
    if ('Age' in k): # if we have exactly two covariates: age/sex ([1/2, 2/5, 2/7] for MIMIC, Code15, BCH)
        tmp = '+ '
    if ('QRS' in k): # MIMIC with MM uses '6'
        tmp = '^ '
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
    
    if (tmp.startswith('+')):
        CovTypes.append('AgeSex')
    elif (tmp.startswith('^')):
        CovTypes.append('Machine')
    else:
        CovTypes.append('None')
        
    if i == 'No Entry':
        CC_or_DS.append('CC')
    else:
        CC_or_DS.append('DS')
    
tmp_df = pd.DataFrame(data=Surv_Model_Types_Combo, columns=['Cov_Surv_Model'])    
df = df.join(tmp_df) 

tmp_df = pd.DataFrame(data=CovTypes, columns=['CovType'])    
df = df.join(tmp_df) 

tmp_df = pd.DataFrame(data=CC_or_DS, columns=['CC_or_DS'])    
df = df.join(tmp_df) 


df = df[df['T99 RS_Conc'] != 'No Entry']


float_keys = ['T99 RS_Conc', 'T5 RS_Conc', 'T1 RS_Conc', 'T2 RS_Conc', 'T10 RS_Conc', 'T1 AUROC', 'T2 AUROC', 'T5 AUROC', 'T10 AUROC', 'T1 AUPRC', 'T2 AUPRC', 'T5 AUPRC', 'T10 AUPRC']
for key in float_keys:
    df[key] = df[key].astype(float)
    
df = df[df['T99 RS_Conc'] > 0.2] # cut models that just totally failed


num_models = 0
for data_folder in ['Code15', 'BCH', 'MIMICIV']:
    tmp_df = df[ (df["Test_Folder"]==data_folder) & ( df["Train_Folder"]==data_folder)]
    num_models += tmp_df.shape[0]
    


cluster_keys = ['Train_Folder', 'Test_Folder', 'Model_Architecture', 'Cov_Surv_Model']

# https://stackoverflow.com/questions/44156051/add-a-series-to-existing-dataframe
# where we're storing everything
counts = df.groupby(cluster_keys).size().to_frame(name='N')
Med = df.groupby(cluster_keys)['T99 RS_Conc'].median().to_frame(name='T99 RS_Conc')
P25 = df.groupby(cluster_keys)['T99 RS_Conc'].quantile(q=0.25).to_frame(name='P25')
P75 = df.groupby(cluster_keys)['T99 RS_Conc'].quantile(q=0.75).to_frame(name='P75')
summary_df = pd.concat([counts, Med, P25, P75], axis=1)

# plots
fig, ax = plt.subplots(3,1, figsize = (3.35,8))


# Code15 with and without covariates
tmp = df[ (df["Test_Folder"]== "Code15")
           * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])


# plt.figure()
ax[0] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='T99 RS_Conc', hue = 'Model_Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[0])
ax[0].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[0].set_title('Code-15')
ax[0].set_ylabel('Concordance')
ax[0].axhline(0.7907, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[0].axvline(3.5,ls='--',color='k')
ax[0].axvline(7.5,ls='--',color='k')
ax[0].axvline(11.5,ls='--',color='k')
ax[0].text(8, 0.78, "Best non-ECG", color='r')
ax[0].set(xlabel=None)
ax[0].legend(loc='lower right')
# ax[0].get_legend().remove()
ax[0].set(xticklabels=[])     

# MIMICIV with and without covariates
tmp = df[ (df["Test_Folder"]== "MIMICIV")
           * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])
# plt.figure()
ax[1] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='T99 RS_Conc', hue = 'Model_Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[1])
ax[1].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[1].set_title('MIMIC-IV')
ax[1].set_ylabel('Concordance')
ax[1].axhline(0.65349, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[1].text(0, 0.66, "Best non-ECG", color='r')
ax[1].axhline(0.728978, ls = '--', color='m') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[1].text(0, 0.718, "Best XGB", color='m')
ax[1].axvline(3.5,ls='--',color='k')
ax[1].axvline(7.5,ls='--',color='k')
ax[1].axvline(11.5,ls='--',color='k')
ax[1].set(xlabel=None)
# ax[1].legend(loc='lower right')  
ax[1].set(xticklabels=[])     
ax[1].get_legend().remove()

# BCH with and without covariates
tmp = df[ (df["Test_Folder"]== "BCH")
           * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin(["InceptionTime", "ResNet"]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])

# plt.figure()
ax[2] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='T99 RS_Conc', hue = 'Model_Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[2])
ax[2].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[2].set_title('BCH')
ax[2].set_ylabel('Concordance')
ax[2].axhline(0.654, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[2].axvline(3.5,ls='--',color='k')
ax[2].axvline(7.5,ls='--',color='k')
ax[2].axvline(11.5,ls='--',color='k')
ax[2].text(0, 0.672, "Best non-ECG", color='r')
ax[2].set(xlabel=None)
# ax[2].legend(loc='lower right')  
ax[2].get_legend().remove()  

plt.tight_layout()
os.makedirs(os.path.join(os.getcwd(),'Pandas Analysis'),exist_ok=True)
plt_loc = os.path.join(os.getcwd(),'Pandas Analysis', 'All-Concordance Figure.pdf')
plt.savefig(plt_loc)

# save out

# Transformer only

fig, ax = plt.subplots(3,1, figsize = (3.35,5))

# BCH with and without covariates
tmp = df[ (df["Test_Folder"]== "BCH")
           * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin(["ECGTransForm"]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])

# plt.figure()
ax[0] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='T99 RS_Conc', hue = 'Model_Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[0])
ax[0].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[0].set_title('BCH')
ax[0].set_ylabel('Concordance')
ax[0].axhline(0.654, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[0].axvline(3.5,ls='--',color='k')
ax[0].axvline(7.5,ls='--',color='k')
ax[0].axvline(11.5,ls='--',color='k')
ax[0].text(0, 0.672, "Best non-ECG", color='r')
ax[0].set(xlabel=None)
# ax[0].legend(loc='lower right')    
ax[0].set(xticklabels=[])     
ax[0].get_legend().remove() 



# Code15 with and without covariates
tmp = df[ (df["Test_Folder"]== "Code15")
           * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin(["ECGTransForm"]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])


# plt.figure()
ax[1] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='T99 RS_Conc', hue = 'Model_Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[1])
ax[1].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[1].set_title('Code-15')
ax[1].set_ylabel('Concordance')
ax[1].axhline(0.7907, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[1].axvline(3.5,ls='--',color='k')
ax[1].axvline(7.5,ls='--',color='k')
ax[1].axvline(11.5,ls='--',color='k')
ax[1].text(6, 0.72, "Best non-ECG", color='r')
ax[1].set(xlabel=None)
# ax[1].legend(loc='lower right') 
ax[1].set(xticklabels=[])     
ax[1].get_legend().remove()    

# MIMICIV with and without covariates
tmp = df[ (df["Test_Folder"]== "MIMICIV")
           * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin(["ECGTransForm"]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])
# plt.figure()
ax[2] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y='T99 RS_Conc', hue = 'Model_Architecture', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[2])
ax[2].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[2].set_title('MIMIC-IV')
ax[2].set_ylabel('Concordance')
ax[2].axhline(0.65349, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[2].text(-0.4, 0.66, "Best non-ECG", color='r')
ax[2].axhline(0.7287, ls = '--', color='m') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[2].text(6, 0.732, "Best XGB", color='m')
ax[2].axvline(3.5,ls='--',color='k')
ax[2].axvline(7.5,ls='--',color='k')
ax[2].axvline(11.5,ls='--',color='k')
ax[2].set(xlabel=None)
ax[2].legend(loc='lower right')  

plt.tight_layout()
os.makedirs(os.path.join(os.getcwd(),'Pandas Analysis'),exist_ok=True)
plt_loc = os.path.join(os.getcwd(),'Pandas Analysis', 'TF All-Concordance Figure.pdf')
plt.savefig(plt_loc)



# %%

# ************ try to include the x-evals

targ_dataset = "InceptionTime"

plt_target = 'T99 RS_Conc'

fig, ax = plt.subplots(3,1, figsize = (3.35,8))
tmp = df[ (df["Test_Folder"]== "Code15")
           # * (df["Train_Folder"]== train_folder)
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin([targ_dataset]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])


# plt.figure()
ax[0] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y=plt_target, hue = 'Train_Folder', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[0])
ax[0].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[0].set_title('Code-15 Test Set')
ax[0].set_ylabel('Concordance')
ax[0].axhline(0.7907, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[0].axvline(3.5,ls='--',color='k')
ax[0].axvline(7.5,ls='--',color='k')
ax[0].axvline(11.5,ls='--',color='k')
ax[0].text(8, 0.755, "Best non-ECG", color='r')
ax[0].set(xlabel=None)
# ax[0].legend(loc='lower right')
ax[0].get_legend().remove()
ax[0].set(xticklabels=[])     
ax[0].set_ylim([0.50,0.85])

# MIMICIV with and without covariates
tmp = df[ (df["Test_Folder"]== "MIMICIV")
           # * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin([targ_dataset]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])
# plt.figure()
ax[1] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y=plt_target, hue = 'Train_Folder', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[1])
ax[1].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[1].set_title('MIMIC-IV Test Set')
ax[1].set_ylabel('Concordance')
ax[1].axhline(0.65349, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[1].text(9, 0.635, "Best non-ECG", color='r')
ax[1].axhline(0.728978, ls = '--', color='m') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[1].text(0, 0.71, "Best XGB", color='m')
ax[1].axvline(3.5,ls='--',color='k')
ax[1].axvline(7.5,ls='--',color='k')
ax[1].axvline(11.5,ls='--',color='k')
ax[1].set(xlabel=None)
# ax[1].legend(loc='lower right')  
ax[1].set(xticklabels=[])     
ax[1].get_legend().remove()
ax[1].set_ylim([0.50,0.85])

# BCH with and without covariates
tmp = df[ (df["Test_Folder"]== "BCH")
           # * (df["Test_Folder"]== df["Train_Folder"])
           # * (dataframe["val_covariate_col_list"].isin(["[2,5]"]))
           # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
            * (df["Model_Architecture"].isin([targ_dataset]))
           ].sort_values(['Cov_Surv_Model','Model_Architecture'])

# plt.figure()
ax[2] = seaborn.boxplot(data=tmp,x='Cov_Surv_Model',y=plt_target, hue = 'Train_Folder', fill = False, order = ['Cla-01','Cla-02','Cla-05','Cla-10','DeepHit','DeepSurv','LH','MTLR','+ Cla-01','+ Cla-02','+ Cla-05','+ Cla-10','+ DeepHit','+ DeepSurv','+ LH','+ MTLR'], ax = ax[2])
ax[2].tick_params(axis='x', rotation=90) # from https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn
ax[2].set_title('BCH Test Set')
ax[2].set_ylabel('Concordance')
ax[2].axhline(0.654, ls = '--', color='r') #https://stackoverflow.com/questions/51891370/draw-a-line-at-specific-position-annotate-a-facetgrid-in-seaborn
ax[2].axvline(3.5,ls='--',color='k')
ax[2].axvline(7.5,ls='--',color='k')
ax[2].axvline(11.5,ls='--',color='k')
ax[2].text(0, 0.65, "Best non-ECG", color='r')
ax[2].set(xlabel=None)
ax[2].legend(loc='lower right')  
ax[2].set_ylim([0.50,0.85])
# ax[2].get_legend().remove()  

plt.tight_layout()
# os.makedirs(os.path.join(os.getcwd(),'Pandas Analysis'),exist_ok=True)
# plt_loc = os.path.join(os.getcwd(),'Pandas Analysis', 'All-Concordance Figure.pdf')
# plt.savefig(plt_loc)

# %% set up for table 3: print out baselines, print out all cross-eval, choose mnually
dataframe = df
    
#  All baselines and cross-evaluations
baselines = {}
for datafolder in ['Code15', 'BCH', 'MIMICIV']:
    tmp = dataframe[ (dataframe["Train_Folder"]== datafolder)
               # * (dataframe["Test_Folder"]!= dataframe["Train_Folder"])
               # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
                * (dataframe["Model_Architecture"].isin(["Feedforward Net","XGB"]))
               ].sort_values(['Cov_Surv_Model','Model_Architecture'])
    
    tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc']
    baselines[datafolder] = tmp2.agg(['median', ("q25", lambda x: pd.Series.quantile(x,0.25)), ("q75", lambda x: pd.Series.quantile(x,0.75)), 'count' ])
    
    # medians =  .median()
    # p25 =  tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc'].quantile(0.25)
    # p75 =  tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc'].quantile(0.75)
    # print(medians)
    

cnns = {}
for datafolder in ['Code15', 'BCH', 'MIMICIV']:
    tmp = dataframe[ (dataframe["Train_Folder"]== datafolder)
               # * (dataframe["Test_Folder"]!= dataframe["Train_Folder"])
               # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
                * (dataframe["Model_Architecture"].isin(["ResNet","InceptionTime"]))
               ].sort_values(['Cov_Surv_Model','Model_Architecture'])
    
    tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc']
    cnns[datafolder] = tmp2.agg(['median', ("q25", lambda x: pd.Series.quantile(x,0.25)), ("q75", lambda x: pd.Series.quantile(x,0.75)),'count'])
    medians =  tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc'].median()
    p25 =  tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc'].quantile(0.25)
    p75 =  tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])['T99 RS_Conc'].quantile(0.75)
    # print(medians)
    

    
    

    
# % Next, build large supplement tables
# this lets you do multiple measures:
# tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])[['T99 RS_Conc','T99 RS_Conc']]
# a = tmp2.agg(['median', ("q25", lambda x: pd.Series.quantile(x,0.25)), ("q75", lambda x: pd.Series.quantile(x,0.75)) ])




# function from Bill
def med_IQR(x):
    print(len(x))
    med = x.median()
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)

    out_txt = f'{med:.4f}' + '('+f'{q25:.4f}'+'-'+f'{q75:.4f}'+')'
    return out_txt

# ST4, concordance censored to year
for datafolder in ['Code15', 'BCH', 'MIMICIV']:
    tmp = dataframe[ (dataframe["Train_Folder"]== datafolder)
               * (dataframe["Test_Folder"]== dataframe["Train_Folder"])
               # * (dataframe["val_covariate_col_list"].isin(["No Entry"]))
                * (dataframe["Model_Architecture"].isin(["ResNet","InceptionTime","XGB","Feedforward Net"]))
               ].sort_values(['Cov_Surv_Model','Model_Architecture'])
    
    tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])[['T1 RS_Conc','T2 RS_Conc','T5 RS_Conc','T10 RS_Conc','T99 RS_Conc']]
    tmp3 = tmp2.agg(['median', ("q25", lambda x: pd.Series.quantile(x,0.25)), ("q75", lambda x: pd.Series.quantile(x,0.75)) ])
    filepath = os.path.join(os.getcwd(),'Pandas Analysis', 'Concordance Table '+datafolder+'.csv')
    tmp3.to_csv(filepath)
    
    tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])[['T1 AUROC','T2 AUROC','T5 AUROC','T10 AUROC', 'T1 AUPRC','T2 AUPRC','T5 AUPRC','T10 AUPRC']]
    tmp3 = tmp2.agg(['median', ("q25", lambda x: pd.Series.quantile(x,0.25)), ("q75", lambda x: pd.Series.quantile(x,0.75)) ])
    filepath = os.path.join(os.getcwd(),'Pandas Analysis', 'AUC Tables '+datafolder+'.csv')
    tmp3.to_csv(filepath)
    
    # tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])[['T1 AUPRC','T2 AUPRC','T5 AUPRC','T10 AUPRC']]
    # tmp3 = tmp2.agg(['median', ("q25", lambda x: pd.Series.quantile(x,0.25)), ("q75", lambda x: pd.Series.quantile(x,0.75)) ])
    # filepath = os.path.join(os.getcwd(),'Pandas Analysis', 'AUPRC Table '+datafolder+'.csv')
    # tmp3.to_csv(filepath)
    
    
    # tmp2 = tmp.groupby(['Train_Folder','Test_Folder','Cov_Surv_Model','Model_Architecture'])[['T1 RS_Conc']]
    # print(tmp2.apply(med_IQR))
    
tmp_bs_df = dataframe.copy()  
T99_Cols = [key for key in tmp_bs_df.columns if 'T99 BS_Conc' in key]
T10_Cols = [key for key in tmp_bs_df.columns if 'T10 BS_Conc' in key]
T2_Cols = [key for key in tmp_bs_df.columns if 'T2 BS_Conc' in key]
T5_Cols = [key for key in tmp_bs_df.columns if 'T5 BS_Conc' in key]
T1_Cols = [key for key in tmp_bs_df.columns if 'T1 BS_Conc' in key]

# %% 05 19 25 extended analysis
ST11 = []
for data_folder_1 in ['Code15', 'BCH', 'MIMICIV']:
    for data_folder_2 in ['Code15', 'BCH', 'MIMICIV']:
        
        Sorted_Cov_CC =   df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['CC_or_DS'].isin(['CC']))  ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        Sorted_NoCov_CC = df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='None')   & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['CC_or_DS'].isin(['CC']))  ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        
        Sorted_Cov_DS =   df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['CC_or_DS'].isin(['DS']))  ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        Sorted_NoCov_DS = df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='None')   & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['CC_or_DS'].isin(['DS']))  ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        
        NoCov_CC_Med = Sorted_NoCov_CC.median()
        NoCov_CC_p25 = Sorted_NoCov_CC.quantile(q=0.25)
        NoCov_CC_p75 = Sorted_NoCov_CC.quantile(q=0.75)
        
        Cov_CC_Med = Sorted_Cov_CC.median()
        Cov_CC_p25 = Sorted_Cov_CC.quantile(q=0.25)
        Cov_CC_p75 = Sorted_Cov_CC.quantile(q=0.75)
        
        NoCov_DS_Med = Sorted_NoCov_DS.median()
        NoCov_DS_p25 = Sorted_NoCov_DS.quantile(q=0.25)
        NoCov_DS_p75 = Sorted_NoCov_DS.quantile(q=0.75)
        
        Cov_DS_Med = Sorted_Cov_DS.median()
        Cov_DS_p25 = Sorted_Cov_DS.quantile(q=0.25)
        Cov_DS_p75 = Sorted_Cov_DS.quantile(q=0.75)
        
        Cov_CCDS_P = mannwhitneyu(Sorted_Cov_CC, Sorted_Cov_DS)
        NoCov_CCDS_P = mannwhitneyu(Sorted_NoCov_CC, Sorted_NoCov_DS)
        
        ST11.append( [ data_folder_1,data_folder_2,'Classifier-Cox', NoCov_CC_Med, NoCov_CC_p25, NoCov_CC_p75, NoCov_CCDS_P[1] ,Cov_CC_Med, Cov_CC_p25, Cov_CC_p75, Cov_CCDS_P[1]] )
        ST11.append( [ data_folder_1,data_folder_2,'DeepSurv',       NoCov_DS_Med, NoCov_DS_p25, NoCov_DS_p75, 0               ,Cov_DS_Med, Cov_DS_p25, Cov_DS_p75, 0] )
        # ST11.append( [ data_folder_1,data_folder_2,'CCDS NoCov P',NoCov_CCDS_P[1], 'CCDS Cov P', Cov_CCDS_P[1],0,0,0] )

ST11pd = pd.DataFrame(ST11)


# ST1.2: InceptionTime vs ResNet, paired
ST12 = []
for data_folder_1 in ['Code15', 'BCH', 'MIMICIV']:
    for data_folder_2 in ['Code15', 'BCH', 'MIMICIV']:
        Sorted_Cov_IT =   df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["InceptionTime"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        Sorted_NoCov_IT = df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='None')   & (df['Model_Architecture'].isin( ["InceptionTime"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        
        Sorted_Cov_RN =   df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["ResNet"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        Sorted_NoCov_RN = df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='None')   & (df['Model_Architecture'].isin( ["ResNet"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        
        
        NoCov_IT_Med = Sorted_NoCov_IT.median()
        NoCov_IT_p25 = Sorted_NoCov_IT.quantile(q=0.25)
        NoCov_IT_p75 = Sorted_NoCov_IT.quantile(q=0.75)
        
        Cov_IT_Med = Sorted_Cov_IT.median()
        Cov_IT_p25 = Sorted_Cov_IT.quantile(q=0.25)
        Cov_IT_p75 = Sorted_Cov_IT.quantile(q=0.75)
        
        NoCov_RN_Med = Sorted_NoCov_RN.median()
        NoCov_RN_p25 = Sorted_NoCov_RN.quantile(q=0.25)
        NoCov_RN_p75 = Sorted_NoCov_RN.quantile(q=0.75)
        
        Cov_RN_Med = Sorted_Cov_RN.median()
        Cov_RN_p25 = Sorted_Cov_RN.quantile(q=0.25)
        Cov_RN_p75 = Sorted_Cov_RN.quantile(q=0.75)
        
        Cov_ITRN_P = wilcoxon(Sorted_Cov_IT, Sorted_Cov_RN)
        NoCov_ITRN_P = wilcoxon(Sorted_NoCov_IT, Sorted_NoCov_RN)
        
        ST12.append( [ data_folder_1, data_folder_2,'IT', NoCov_IT_Med, NoCov_IT_p25, NoCov_IT_p75,NoCov_ITRN_P[1], Cov_IT_Med, Cov_IT_p25, Cov_IT_p75, Cov_ITRN_P[1]] )
        ST12.append( [ data_folder_1, data_folder_2,'RN', NoCov_RN_Med, NoCov_RN_p25, NoCov_RN_p75,0              , Cov_RN_Med, Cov_RN_p25, Cov_RN_p75, 0] )
        # ST12.append( [ data_folder+'ITRN NoCov P',NoCov_ITRN_P[1], data_folder+'ITRN Cov P', Cov_ITRN_P[1]] )
    
ST12pd = pd.DataFrame(ST12)
    

# ST2: PersonR for Concordance ~ horizon
ST2 = []
for data_folder in ['Code15', 'BCH', 'MIMICIV']:
    Sorted_Cov_Conc =   df[ (df["Test_Folder"]==data_folder) & ( df["Train_Folder"]==data_folder) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['horizon'].isin(["1.0","2.0","5.0","10.0"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
    Sorted_Cov_Horiz =   df[ (df["Test_Folder"]==data_folder) & ( df["Train_Folder"]==data_folder) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['horizon'].isin(["1.0","2.0","5.0","10.0"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['horizon'].astype('float')
    a,b = pearsonr(Sorted_Cov_Conc,Sorted_Cov_Horiz)
    
    Sorted_NoCov_Conc =   df[ (df["Test_Folder"]==data_folder) & ( df["Train_Folder"]==data_folder) & (df['CovType']=='None') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['horizon'].isin(["1.0","2.0","5.0","10.0"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
    Sorted_NoCov_Horiz =   df[ (df["Test_Folder"]==data_folder) & ( df["Train_Folder"]==data_folder) & (df['CovType']=='None') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) & (df['horizon'].isin(["1.0","2.0","5.0","10.0"]) )   ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['horizon'].astype('float')
    c,d = pearsonr(Sorted_NoCov_Conc,Sorted_NoCov_Horiz)
    
    ST2.append( [ data_folder+'Cov Horiz~Conc', a, b, data_folder+'NoCov Horiz~Conc', c, d] )
    
ST2pd = pd.DataFrame(ST2)    

    
# ST3: Do demographics help? paired.
ST3 = []
for data_folder_1 in ['Code15', 'BCH', 'MIMICIV']:
    for data_folder_2 in ['Code15', 'BCH', 'MIMICIV']:
            
        Sorted_Cov_Conc =   df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='AgeSex') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        Sorted_NoCov_Conc =   df[ (df["Test_Folder"]==data_folder_2) & ( df["Train_Folder"]==data_folder_1) & (df['CovType']=='None') & (df['Model_Architecture'].isin( ["InceptionTime", "ResNet"]) ) ].sort_values(['Train_Folder', 'Cov_Surv_Model','Model_Architecture','Rand_Seed'])['T99 RS_Conc']
        
        a = Sorted_Cov_Conc.median()
        b = Sorted_Cov_Conc.quantile(q=0.25)
        c = Sorted_Cov_Conc.quantile(q=0.75)
        
        d = Sorted_NoCov_Conc.median()
        e = Sorted_NoCov_Conc.quantile(q=0.25)
        f = Sorted_NoCov_Conc.quantile(q=0.75)
        
        g = wilcoxon(Sorted_Cov_Conc,Sorted_NoCov_Conc)
        
        ST3.append( [ data_folder_1, data_folder_2, 'Cov', a, b, c, 'NoCov', d, e, f, 'P=',g[1]] )
         
ST3pd = pd.DataFrame(ST3)