# -*- coding: utf-8 -*-
"""
Written by Mingxuan Liu
(minor modifications: PVL)
# 09/12/24L discharge time no longer records admit time (only affects multimodal)
"""

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
from datetime import datetime, timedelta
import os

# from tableone import TableOne

# %% Data paths

patient_csv_dir = "/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimiciv/2.2/hosp/patients.csv.gz"
# patient_csv_dir = os.path.join(os.getcwd(), 'MIMIC IV', 'patients.csv.gz')

admission_csv_dir = "/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimiciv/2.2/hosp/admissions.csv.gz"
# admission_csv_dir = os.path.join(os.getcwd(), 'MIMIC IV', 'admissions.csv.gz')

dat_ecg = pd.read_csv("/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv")
# dat_ecg = pd.read_csv(os.path.join(os.getcwd(), 'MIMIC IV', "machine_measurements.csv"))

dat_record = pd.read_csv("/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv")
# dat_record = pd.read_csv(os.path.join(os.getcwd(), 'MIMIC IV', "record_list.csv"))


dat_dic = pd.read_csv("/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimic-iv-ecg/1.0/machine_measurements_data_dictionary.csv")
# dat_dic = pd.read_csv(os.path.join(os.getcwd(), 'MIMIC IV', "machine_measurements_data_dictionary.csv"))

dat_note = pd.read_csv("/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimic-iv-ecg/1.0/waveform_note_links.csv")
# dat_note = pd.read_csv(os.path.join(os.getcwd(), 'MIMIC IV', "waveform_note_links.csv"))


# %% Extract DoD from patients table
dat_dod = pd.read_csv(patient_csv_dir)

print(pd.isna(dat_dod.dod).mean())
print(f"#patients in patients table: {len(dat_dod.subject_id.unique())}")

dat_dod["dod"] = [datetime.strptime(t, "%Y-%m-%d") if not pd.isna(t) else t for t in dat_dod["dod"]]
dat_dod.head()
#patients in patients table: 299712

# %% extract the max dischtime from each patient from admissions table

dat_hosp = pd.read_csv(admission_csv_dir)[["subject_id", "hadm_id", "admittime","dischtime", "deathtime"]]
print(dat_hosp.head())
print(f"#patients in hospotal table: {len(dat_hosp.subject_id.unique())}")

dat_hosp["admittime"] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in dat_hosp["admittime"]] # added PVL 07 29 24
dat_hosp["dischtime"] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in dat_hosp["dischtime"]]
max_disch_time = dat_hosp.groupby("subject_id").apply(lambda x: max(x.dischtime)).to_frame() # Include_Groups arg is deprecated PVL 6/12/24
max_disch_time.columns = ["max_disch_time"]
max_disch_time = max_disch_time.reset_index()

max_disch_time

# %% ecg table with machine measurements
# ecg_dir = "/lab-share/CHIP-Lacava-e2/Public/physionet.org/files/mimic-iv-ecg/1.0"


print(dat_ecg.shape)
print(f"#patients in ecg table: {len(dat_ecg.subject_id.unique())}")
dat_ecg.head()


dat_ecg["ecg_time"] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in dat_ecg["ecg_time"]]
dat_ecg["ecg_date"] = [t.strftime("%Y-%m-%d") for t in dat_ecg["ecg_time"]]

max_ecg_time = dat_ecg.groupby("subject_id").apply(lambda x: max(x.ecg_time)).to_frame() # Include_Groups arg is deprecated PVL 6/12/24
max_ecg_time.columns = ["max_ecg_time"]
max_ecg_time = max_ecg_time.reset_index()
max_ecg_time

min_ecg_time = dat_ecg.groupby("subject_id").apply(lambda x: min(x.ecg_time)).to_frame() # Include_Groups arg is deprecated PVL 6/12/24
min_ecg_time.columns = ["min_ecg_time"]
min_ecg_time = min_ecg_time.reset_index()
min_ecg_time

pd.merge(min_ecg_time, max_ecg_time, on="subject_id")

# check the most weird patient
max_ecg_patient = (max_ecg_time.loc[:, "max_ecg_time"] - min_ecg_time.loc[:, "min_ecg_time"]).argmax()
pd.merge(min_ecg_time, max_ecg_time, on="subject_id").iloc[max_ecg_patient, :]

dat_ecg.loc[dat_ecg.subject_id == 19822093, :]

tmp = dat_ecg.groupby("subject_id")["ecg_time"].apply(lambda x: x.max() - x.min())
tmp = [t.days for t in tmp]
np.max(tmp) / 365



# %% recrod list

dat_record["ecg_time"] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in dat_record["ecg_time"]]
dat_record["ecg_date"] = [t.strftime("%Y-%m-%d") for t in dat_record["ecg_time"]]

print(f"#patients in ecg record table: {len(dat_record.subject_id.unique())}")
dat_record

tmp = dat_record.groupby("subject_id")["ecg_time"].apply(lambda x: x.max()-x.min())
tmp = [t.days/365 for t in tmp]
print(f"maxmium ecg record follow-up time in years: {np.max(tmp)}")

dat_record.loc[dat_record.subject_id == 19822093, :] # seems the information does not match with that in measurements

# %% use ecg time from the records
max_ecg_time = dat_record.groupby("subject_id").apply(lambda x: max(x.ecg_time)).to_frame() # Include_Groups arg is deprecated PVL 6/12/24
max_ecg_time.columns = ["max_ecg_time"]
max_ecg_time = max_ecg_time.reset_index()
max_ecg_time

min_ecg_time = dat_record.groupby("subject_id").apply(lambda x: min(x.ecg_time)).to_frame() # Include_Groups arg is deprecated PVL 6/12/24
min_ecg_time.columns = ["min_ecg_time"]
min_ecg_time = min_ecg_time.reset_index()
min_ecg_time

pd.merge(min_ecg_time, max_ecg_time, on="subject_id")

dat = pd.merge(dat_ecg, dat_record, on=["subject_id", "study_id"])
dat = dat.rename(columns={"ecg_time_y":"ecg_time", "ecg_date_y":"ecg_date"})
print(dat.shape)
dat.head()


dat_note["charttime"] = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in dat_note["charttime"]]
print(f"#patients in ecg note table: {len(dat_note.subject_id.unique())}")
dat_note.head()

tmp = dat_note.groupby("subject_id")["charttime"].apply(lambda x: x.max()-x.min())
tmp = [t.days/365 for t in tmp]
print(f"maxmium ecg note follow-up time in years: {np.max(tmp)}")

dat_note.groupby("subject_id").count()["waveform_path"].hist(bins=100)

patient_ids = dat_note.subject_id.unique()
patient_ids_demo = patient_ids[dat_note.groupby("subject_id").count()["waveform_path"] > 200]
patient_ids_demo = dat_note.loc[dat_note.subject_id == patient_ids_demo[0], :]

last_time = patient_ids_demo["charttime"].max()
first_time = patient_ids_demo["charttime"].min()
duration = last_time - first_time
print(f"duration of the patient with subject_id {patient_ids_demo.subject_id.unique()}: {duration}")
patient_ids_demo

# %% Merge

dat = pd.merge(dat, dat_dod, how="inner", on="subject_id")
print(len(dat.subject_id.unique()))
dat = pd.merge(dat, max_ecg_time, how="inner", on="subject_id")
print(len(dat.subject_id.unique()))
dat = pd.merge(dat, max_disch_time, how="left", on="subject_id")
# print(len(dat.subject_id.unique()))
# dat = pd.merge(dat, dat_hosp, how="left", on="subject_id")
print(dat.shape)
print(len(dat.subject_id.unique()))
dat.head()

dat.to_csv(os.path.join(os.getcwd(), 'MIMIC IV', "machine_measurements_with_dod.csv"))

dat.head()

tmp = pd.merge(max_disch_time, max_ecg_time, how="inner", on="subject_id")
# % of patients having ecg after discharged
print(np.mean(tmp["max_disch_time"] < tmp["max_ecg_time"])) 

# %% add event and time-to-event to "machine_measurements_with_dod.csv"
dat["event"] = ~pd.isna(dat["dod"])
ecg_followup = dat["max_ecg_time"] - dat["ecg_time"] 
inhosp_followup = dat["max_disch_time"] - dat["ecg_time"]

dod_followup = [dat["dod"][i] - datetime.strptime(dat["ecg_date"][i], "%Y-%m-%d") for i in range(dat.shape[0])] # no time for dod, only date
tmp_dod_followup = [t.days/365 for t in dod_followup if not pd.isna(t)]
plt.hist(tmp_dod_followup, bins=100)
print(f"the maximum dod follow-up time in year:{np.max(tmp_dod_followup)}")

tmp_inhosp_followup = [t.days/365 for t in inhosp_followup if not pd.isna(t)]
plt.hist(tmp_inhosp_followup, bins=100)
print(f"the maximum inhosp follow-up time in year:{np.max(tmp_inhosp_followup)}")

tmp_ecg_followup = [t.days/365 for t in ecg_followup if not pd.isna(t)]
plt.hist(tmp_ecg_followup, bins=100)
print(f"the maximum ecg follow-up time in year:{np.max(tmp_ecg_followup)}")

# note: patients who are survival according to dod may lack of the records about max disch time, 
# or the max_disch_time is far earlier than the ecg time, thus using max_ecg_time instead. 
# From MIMIC-IV website: "If the individual survived for at least one year after their last hospital 
# discharge, then the dod column will have a NULL value."

# note from the document of "mimic-iv-ecg" (the User notes section): "the ECG time stamps
#  could be significantly out of sync with the corresponding time stamps in the MIMIC-IV 
#  Clinical Database, MIMIC-IV Waveform Database, or other modules in MIMIC-IV. An additional 
#  limitation, as noted above, is that some of the ECGs provided here were collected outside of
#  the ED and ICU. This means that the timestamps for those ECGs won't overlap with data from 
#  the MIMIC-IV Clinical Database."

dat["time_to_event"] = [dat.loc[i, "max_disch_time"] - dat.loc[i, "ecg_time"] + timedelta(days=365) if pd.isna(dat.loc[i, "dod"]) else dod_followup[i] for i in range(dat.shape[0])] 
dat["time_to_event"] = [dat["max_ecg_time"][i] - dat["ecg_time"][i] + timedelta(days=365) if pd.isna(dat["time_to_event"][i]) or dat["time_to_event"][i].days < 0 else dat["time_to_event"][i] for i in range(dat.shape[0])] 
dat["time_to_event"] = [t.days for t in dat["time_to_event"]]

print(f"the maximum time-to-event in year is: {(dat['time_to_event'] /365).max()}")
(dat["time_to_event"] /365).hist(bins=200)

tmp_0 = dat.loc[dat["time_to_event"] == 0, :]
tmp_0.loc[pd.isna(tmp_0["dod"]), ]

# %% add info on admission start/end time if admitted - PVL 07 29 24. 
# Takes ~13 min.
# from tqdm import tqdm
admit_start = []
admit_end = []
# for s_id, ecg_t in tqdm(zip(dat['subject_id'],dat['ecg_time_x']), total = len(dat['subject_id'])):
for s_id, ecg_t in zip(dat['subject_id'],dat['ecg_time_x']):
    
    inds = (dat_hosp["subject_id"]==s_id)
    
    admits = dat_hosp["admittime"][inds]
    dischs = dat_hosp["dischtime"][inds]
    
    # mark the admission start time if ECG was taken while admitted, else add 'Not Admitted'
    appending_admit = 'Not Admitted'
    appending_disch = 'Not Admitted'
    for start,stop in zip(admits, dischs):
        if (ecg_t > start) and (ecg_t < stop):
            appending_admit =  str(start)
            appending_disch =  str(stop)
            break
    admit_start.append(appending_admit)
    admit_end.append(appending_disch)
    
dat["admit_time_before_ecg"] = admit_start
dat["disch_time_after_ecg"] = admit_end


# %%




dat.to_csv(os.path.join(os.getcwd(), 'MIMIC IV', "machine_measurements_survival_sept2024.csv"))

# import pickle
# with open('machine_measurements_survival.pkl', 'wb') as file:
#     pickle.dump(dat, file)
    
# %% information extraction
# dat["anchor_age"].hist()
# dat['Follow-up Time'] = dat["time_to_event"]
# dat["Sex"] = dat["gender"]
# dat["Death"] = dat["event"]

