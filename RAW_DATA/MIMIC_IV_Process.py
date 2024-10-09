# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:33:00 2024

"""

import os
import numpy as np
import scipy.signal as sig
import numpy as np
import pandas as pd

import h5py
import torch # to set manual seed
from torch.utils.data import random_split
import wfdb # wfdb?


data_dir = os.path.join(os.getcwd(), 'MIMIC IV') # adjust

# Adjust ECGs to match Code-15 standard
def adjust_sig(signal):
    # MIMIC IV signals are 10s at 500Hz, 5k total samples
    # we seek 10s @ 400Hz, centered, padded to 4096.
    # so resample to 4k samples
    # then pad with 48 0's on both sides of axis=0
    new_sig = sig.resample(signal, 4000, axis=0)
    new_sig = np.pad(new_sig, ((48,48),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig
    


file_path = os.path.join(os.getcwd(), 'MIMIC IV', 'machine_measurements_survival.csv')
# file_path = '//lab-share//CHIP-Lacava-e2//Public//physionet.org//files//mimic-iv-ecg//1.0//machine_measurements_survival.csv'

record_list = pd.read_csv(file_path, low_memory=False)

ecg_dir = os.path.join(os.getcwd(), 'MIMIC IV', 'ECG') # where ECG lives
# ecg_dir = '//lab-share//CHIP-Lacava-e2//Public//physionet.org//files//mimic-iv-ecg//1.0//files'

import time

# Parse all files, assemble them into dictionaries
Dat_Dict = {}
PID_List = []
# a = time.time()
for PID,SID,TTE,Event in zip(record_list['subject_id'],record_list['study_id'],record_list['time_to_event'],record_list['event']) :
    
    P4 = 'p' + str(PID)[:4] # used in file name
    PID = 'p' + str(PID)
    SID2 = str(SID)
    SID = 's' + str(SID)
    rec_path = f'{ecg_dir}/{P4}/{PID}/{SID}/{SID2}' 
    rd_record = wfdb.rdrecord(rec_path)
    
    signal = rd_record.p_signal # 5k x 12 signal
    
    # filter out nans
    if (np.isnan(signal).any()):
        continue


    # ready to store
    if PID not in Dat_Dict.keys():
        Dat_Dict[PID] = {}
        PID_List.append(PID)
        b = time.time()
        # time_taken = b-a
        # a = b
        # print('PID num: ' +  str(len(PID_List)) + ' t: ' + str(time_taken))
        
    if SID not in Dat_Dict[PID].keys():
        Dat_Dict[PID][SID] = {}
        Dat_Dict[PID][SID]['Signal'] = adjust_sig(signal) # store adjusted signal
        
        # time to event is in days, convert to years
        TTE = float(TTE) /365 # adjust TTE. sometimes get '0' for same-day, so place a lower value of 0.5 /365.
        if (TTE < 1e-4):
            TTE = 0.5 / 365
            
        Dat_Dict[PID][SID]['TTE'] = TTE     # float
        Dat_Dict[PID][SID]['Event'] = Event # bool
        
    # rd_record = wfdb.rdrecord(rec_path) 
    # wfdb.plot_wfdb(record=rd_record, figsize=(124,18), title='Study 41420867 example', ecg_grids='all')

print('debug. # entires: ' + str(len(PID_List)))

# now we split the PID list randomly. 20% test, 80% train.
Te_Count = int(len(PID_List) * 0.2)
Tr_Count = len(PID_List) - Te_Count

torch.manual_seed(12345)
PID_Index_Train, PID_Index_Test = random_split(range(len(PID_List)), [Tr_Count, Te_Count])   


# per https://stackoverflow.com/questions/58089499/how-to-combine-many-numpy-arrays-efficiently
# add everything to list, then concatenate at end

# now we assemble. lists, then numpy, then hdf5.
Train_X = []
Train_Y = []
for i in PID_Index_Train:
    PID = PID_List[i]
    for SID in Dat_Dict[PID].keys():
        Train_X.append(Dat_Dict[PID][SID]['Signal'])
        Event = 0
        if Dat_Dict[PID][SID]['Event'] == True:
            Event = 1
        Train_Y.append( [ PID[1:], SID[1:], 1, Dat_Dict[PID][SID]['TTE'], Event] )    
        # Train_Y.append( [ float(PID[1:]), float(SID[1:]), float(1), float(Dat_Dict[PID][SID]['TTE']), float(Event)] )
       
        
Train_X = np.stack(Train_X, axis=0).astype(np.float32)
Train_Y = np.stack(Train_Y, axis=0).astype(np.float32)

# now we assemble. lists, then numpy, then hdf5.
Test_X = []
Test_Y = []
for i in PID_Index_Test:
    PID = PID_List[i]
    for SID in Dat_Dict[PID].keys():
        Test_X.append(Dat_Dict[PID][SID]['Signal'])
        Event = 0
        if Dat_Dict[PID][SID]['Event'] == True:
            Event = 1
        Test_Y.append( [ PID[1:], SID[1:], 1, Dat_Dict[PID][SID]['TTE'],Event] )
        
Test_X = np.stack(Test_X, axis=0).astype(np.float32)
Test_Y = np.stack(Test_Y, axis=0).astype(np.float32)

# Save out to a numpy array
column_names = ['PID', 'SID', 'Placeholder', 'TTE', 'Event']

Temp = os.path.join(os.getcwd(),'MIMICIV_Processed')
if (os.path.exists(Temp) == False):
    os.mkdir(Temp)
    
Temp = os.path.join(os.getcwd(),'MIMICIV_Processed','TRAIN_DATA') 
if (os.path.exists(Temp) == False):
    os.mkdir(Temp)
    
Temp = os.path.join(os.getcwd(),'MIMICIV_Processed','TEST_DATA')
if (os.path.exists(Temp) == False):
    os.mkdir(Temp)

Out_Path_Train = os.path.join(os.getcwd(),'MIMICIV_Processed', 'TRAIN_DATA')
Out_Path_Test  = os.path.join(os.getcwd(),'MIMICIV_Processed', 'TEST_DATA')

with h5py.File(os.path.join(Out_Path_Train,'Train_Data.hdf5'), "w") as f:
    f.create_dataset('column_names', data = column_names)
    f.create_dataset('x',       data = Train_X)
    f.create_dataset('y',       data = Train_Y)
    
with h5py.File(os.path.join(Out_Path_Test,'Test_Data.hdf5'), "w") as f:
    f.create_dataset('column_names', data = column_names)
    f.create_dataset('x',       data = Test_X)
    f.create_dataset('y',       data = Test_Y)




