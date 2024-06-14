# -*- coding: utf-8 -*-
"""
This file reads in raw Code-15 data from ./RAW_DATA/Cdoe15/TRAIN
And saves out processed data (Numpy Arrays) for downstream work to ./RAW_DATA/Code15_Processed/TRAIN_DATA and ./RAW_DATA/Code15_Processed/TEST_DATA


It also works for data subsets (if you only include several "exams_part{N}.hdf5" of code15)
[Likely has pieces borrowed from stackoverflow or online tutorials]
"""
print('this version of the script only keeps ECG entries with TTE and E explicitly stated (none are inferred)')

import numpy as np
import h5py
import os
import csv
import time
import torch # to set manual seed
from torch.utils.data import random_split

# Uncomment next two lines to run from e.g. Spyder
# import collections
# collections.Callable = collections.abc.Callable

#%% Data directories
Code15_ECG_Directory = os.path.join(os.getcwd(),'RAW_DATA','Code15','TRAIN')
Code15_exams_loc = os.path.join(os.getcwd(),'RAW_DATA','Code15','exams.csv')


Out_Path_Train = os.path.join(os.getcwd(),'RAW_DATA','Code15_Processed', 'TRAIN_DATA')
Out_Path_Test  = os.path.join(os.getcwd(),'RAW_DATA','Code15_Processed', 'TEST_DATA')


Temp = os.path.join(os.getcwd(),'RAW_DATA','Code15_Processed')
if (os.path.exists(Temp) == False):
    os.mkdir(Temp)
    
Temp = os.path.join(os.getcwd(),'RAW_DATA','Code15_Processed','TRAIN_DATA') 
if (os.path.exists(Temp) == False):
    os.mkdir(Temp)
    
Temp = os.path.join(os.getcwd(),'RAW_DATA','Code15_Processed','TEST_DATA')
if (os.path.exists(Temp) == False):
    os.mkdir(Temp)




# %% 
# Grab data placed in 'train' folder
train_exam_id = []
train_tracings = []
for f in os.listdir(Code15_ECG_Directory): # load all hdf5
    print(f)
    with h5py.File(os.path.join(Code15_ECG_Directory,f), "r") as h:
        print("Keys: %s" % h.keys())
        exam_id = h['exam_id'][()]
        tracings = h['tracings'][()]
        train_exam_id.append(exam_id[:-1])        # the last one is just '0'
        # print(tracings[-1])
        train_tracings.append(tracings[:-1])      # ... so cut it
        
train_exam_id = np.hstack([k for k in train_exam_id])
train_x = np.vstack([k for k in train_tracings]) 
del exam_id
del tracings
del train_tracings
    
# and now grab the csv
with open(Code15_exams_loc) as f:
    reader = csv.reader(f, delimiter=",")   
    Values = np.array(list(reader))
    f.close()



# %% Turn that into a dict; per PID and Exam_ID I want to have the ECG and all other measures
Dat_Dict = {}
for row in Values[1:]:
    SID     = int(row[0])
    age     = int(row[1])
    is_male = bool(row[2]=='True')
    # nn_predct = float(row[3])
    m_1dAVb = bool(row[4]=='True') # 'm' signifies 'measure'
    RBBB    = bool(row[5]=='True')
    LBBB    = bool(row[6]=='True')
    SB      = bool(row[7]=='True')
    ST      = bool(row[8]=='True')
    AF      = bool(row[9]=='True')
    
    PID     = int(row[10])
    
    tmp = row[11]
    if (tmp == ''):
        death = -1 # unmarked - infer from other cases or trash sample    
    else:
        death   = bool(row[11]=='True')
    
    tmp = row[12]
    if (row[12] == ''):
        continue # in the version of the script, just skip these entries
        # time_y = -1.0 # time to event unmarked - infer from other cases or trash sample
    else:
        time_y  = float(row[12])
        if ( abs(time_y) < 1e-4):
            time_y = 1.0 / 365.0
            
        
    norm_ecg= bool(row[13]=='True')
    
    tmp = np.where(train_exam_id == SID)[0]
    if (len(tmp) == 1):
        trace_index = tmp[0]
        trace = train_x[trace_index]
    else:
        continue
    
    
    if (PID) not in Dat_Dict.keys():
        Dat_Dict[PID] = {}
    
    if (SID) not in Dat_Dict[PID].keys():
        Dat_Dict[PID][SID] = {}
        
        Dat_Dict[PID][SID]['ECG'] = trace
        Dat_Dict[PID][SID]['age'] = age
        Dat_Dict[PID][SID]['event'] = death
        Dat_Dict[PID][SID]['tte'] = time_y
        Dat_Dict[PID][SID]['other'] = [is_male, m_1dAVb, RBBB, LBBB, SB, ST, AF, norm_ecg]

# %% divide data by PID
PID_List = [k for k in Dat_Dict.keys()]
Te_Count = int(len(PID_List) * 0.2)
Tr_Count = len(PID_List) - Te_Count



torch.manual_seed(12345)
PID_Index_Train, PID_Index_Test = random_split(range(len(PID_List)), [Tr_Count, Te_Count])   

# Now we turn everything into lists before assembling them
Tr_List_X = []
Tr_List_Y = []
for i in PID_Index_Train:
    PID = PID_List[i]
    for SID in Dat_Dict[PID]: # should just be one
        a = Dat_Dict[PID][SID]
    
        Tr_List_X.append(a['ECG'])
        tmp = [PID,SID,a['age'],a['tte'],a['event']] +  [k for k in a['other']]
        Tr_List_Y.append(tmp)
        
Te_List_X = []
Te_List_Y = []
for i in PID_Index_Test:
    PID = PID_List[i]
    for SID in Dat_Dict[PID]: # should just be one
        a = Dat_Dict[PID][SID]
    
        Te_List_X.append(a['ECG'])
        tmp = [PID,SID,a['age'],a['tte'],a['event']] +  [k for k in a['other']]
        Te_List_Y.append(tmp)
        
# and convert that to numpy
Test_X = np.stack(Te_List_X, axis=0).astype(np.float32)
Test_Y = np.stack(Te_List_Y, axis=0).astype(np.float32)

Train_X = np.stack(Tr_List_X, axis=0).astype(np.float32)
Train_Y = np.stack(Tr_List_Y, axis=0).astype(np.float32)

column_names = ['PID', 'ECG_ID', 'age', 'time_to_event', 'event', 'is_male', 'm_1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'norm_ecg']
for i,k in enumerate(column_names):
    print (i,k)

# %% Now save out the training data and testing data as numpy arrays


with h5py.File(os.path.join(Out_Path_Train,'Train_Data.hdf5'), "w") as f:
    f.create_dataset('column_names', data = column_names)
    f.create_dataset('x',       data = Train_X)
    f.create_dataset('y',       data = Train_Y)
    
with h5py.File(os.path.join(Out_Path_Test,'Test_Data.hdf5'), "w") as f:
    f.create_dataset('column_names', data = column_names)
    f.create_dataset('x',       data = Test_X)
    f.create_dataset('y',       data = Test_Y)
