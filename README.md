# ecg-survival-benchmark


## Quick Start
Build a singularity container from Sing_Torch_05032024.def

Minimal Code-15 data:
Go to       https://zenodo.org/records/4916206
Download    exams.csv                                  to ./RAW_DATA/Code15/exams.csv
Download    exams_part0.hdf5                           to ./RAW_DATA/Code15/TRAIN/exams_part0.hdf5
Run         ./RAW_DATA/Code15_ECG_Process_8020.py      with os.getcwd() as ./ 
Move        ./RAW_DATA/Code15_Processed                to ./HDF5_DATA/Code15
---
Open 	    model_runner_caller.py
Run         top section

You should see:
1) a model appear in ./Trained Models/

## Installation
We ran everything using a singularity container built from from Sing_Torch_05032024.def

## Setting up Datasets
Code-15 (100GB RAM, ~2 hours)
Download data from       https://zenodo.org/records/4916206
Place all .hdf5 in       ./RAW_DATA/Code15/TRAIN/exams_part0…17.hdf5
Place exams.csv in       ./RAW_DATA/Code15/exams.csv
Run                      ./RAW_DATA/Code15_ECG_Process_8020.py          with os.getcwd() as ./ 
Move                     ./RAW_DATA/Code15_Processed/…                  to     ./HDF5_DATA/Code15

MIMIC-IV (300GB RAM, ~8 hours)
From “hosp” in  "https://physionet.org/content/mimiciv/2.2/  
	Download /patients.csv.gz         to ./RAW_DATA/MIMIC IV/
	Download /admissions.csv.gz       to ./RAW_DATA/MIMIC IV/

From https://physionet.org/content/mimic-iv-ecg/1.0/ 
	Download machine_measurements.csv                   to ./RAW_DATA/MIMIC IV/ 
	Download machine_measurements_data_dictionary.csv   to ./RAW_DATA/MIMIC IV/
	Download record_list.csv                            to ./RAW_DATA/MIMIC IV/
	Download waveform_note-links.csv                    to ./RAW_DATA/MIMIC IV/
	Download “files”                                    to ./RAW_DATA/MIMIC IV/ECGs/p1000….
Run 				                            ./RAW_DATA/MIMIC_IV_PreProcess.py
Run .                                                       ./RAW_DATA/MIMIC_IV_Process.py          with os.getcwd() as ./RAW_DATA
Move                                                        ./RAW_DATA/MIMICIV_Processed/…          to ./HDF5_DATA/MIMICIV



## Use Overview:

Classifier-based survival models are built and evaluated through Model_Runner_SurvClass
PyCox survival models are built and evaluated through Model_Runner_PyCox

To manually build/evaluate models, use Model_Runner_Caller.py, which is pre-set with minimal calls to train a model.

To do this with job files, structure the args to Model_Runner_SurvClass/PyCox similarly to Model_Runner_Caller does it (more on this later).

Once you have several trained models, you can summarize them to tables using Summarize_Trained_Models.py
This creates the Summary_Tables folder, which includes: 
Trained_Model_Summary.csv (containing all training arguments, default arguments, and model metrics), 
Trained_Model_Main_Results_Table.csv (AUROC / AUPRC / all-time Brier and Concordance, averaged over all random seeds), 
Trained_Model_BrierConcordance_Table.csv (Brier and Concordance measures, averaged over all random seeds, right-censored to several time limits), 
Trained_Model_BS_BrierConcordance_Table.csv (Brier and Concordance measures, averaged over all bootstraps of all random seeds, right-censored ot several time limits), and Trained_Model_CrossvalTable.csv (1-yr AUROC / AUPRC, all-time Brier and Concordance, averaged over all random seeds)

Kaplan-Meier and histogram Survival Functions can be made by adapting Gen_KM_Curves.py

We include Job_File_Maker_SurvClass and Job_File_Maker_PyCox - these generate job files in our institution's format, but can be adapted to yours.
To generate job files based on trained models, use Job_File_Maker_From_Summary.py, after summarizing trained models with Summarize_Trained_Models.py
Job_File_Maker_From_Summary.py can be adapted to add / remove / change model runner arguments (ex: switch the evaluation dataset)

---
Model_Runner_SurvClass/PyCox: Everything is interpreted via a series of string-string argument-value pairs. Arguments must begin with "--" and be separated by spaces. These are turned into a dictionary mapping arguments to values.

Model_Runner_SurvClass initializes a specific_model, which is a generic_model. Each script (Runner, then generic_model, then specific_model) interprets the arguments dictionary.
Model_Runner_PyCox initializes a specific_model, which is a generic_pycox_model, which is a generic_model. Each script (Runner, then generic_model, then generic_pycox_model, then specific_model) interprets the arguments dictionary. 
This format allows flexibility in adding new model types, since e.g. managing training is done by a more general class.
Models are evaluated after training, but a model can be evaluated without training by loading the highest-validation-metric model with "--Load Best" and skipping training with "--Epoch_End -1"

Model_Runner_SurvClass/PyCox loads data and splits it into ECG Data['x_train'], Data['x_test'], Data['x_valid'], and Label(y)-equivalents. Then it more-or-less calls model.init(), model.load() (if applicable), model.fit(), and model.eval(), then sets up output folders, evaluates the models, and saves plots, .csvs, and .hdf5s with results.

Since models are evaluated on-the-fly, a GPU is required for both training and evaluation.


## Example Job File:
See Job_File_Maker_PyCox.py 

## On Data formatting:
Patient ID is currently assumed to to be Data['y_train'][:,0] . This is currently hard-coded but can be easily modified to a passed arg
We typically put ECG_ID in [:,1] (this doesn't matter), age in [:,2] (again, doesn't matter), 'time_to_event' in [:,3], and 'event' in [:,4]. time_to_event and event columns are passed args for both train and test.

We tried to make data processing flexible: inputs in the N-H (single-channel time series) or N-H-W (multi-channel time series) format are expanded to the image N-C-H-W format. Individual modelsl then change the shape as needed (an image model stays in N-C-H-W, ECG models tend to prefer N-H-W).


"""

