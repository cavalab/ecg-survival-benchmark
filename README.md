# ecg-survival-benchmark

This is a set of scripts for modeling patient mortality for the Code-15 and MIMIC-IV datasets in support of https://arxiv.org/abs/2406.17002
The scripts include data processing, model building (deep survival models and classifiers), survival function building (for classifiers), and evaluating models to .csv files.
We use PyCox deep survival models (LH, MTLR, CoxPH (DeepSurv), and DeepHit).
We also implement classifiers predicting mortality by a time horizon (e.g. --horizon 10.0), assuming censored patients survive, and then feed those classifiers through Cox regressions.
Evaluations are based on Concordance, Brier Score, AUPRC, and AUROC. 

## Quick Start

Build a singularity container from `Sing_Torch_05032024.def`

Minimal Code-15 data:

1. Go to       https://zenodo.org/records/4916206
1.  Download    `exams.csv`                                  to `./RAW_DATA/Code15/exams.csv`
1. Download    `exams_part0.hdf5`                           to `./RAW_DATA/Code15/TRAIN/exams_part0.hdf5`
1. Run         `./RAW_DATA/Code15_ECG_Process_8020.py`      with `os.getcwd()` as ./ 
1. Move        `./RAW_DATA/Code15_Processed`                to `./HDF5_DATA/Code15`
1. Open        `model_runner_caller.py`
1. Run         top section

You should see a model folder, populated with .pdfs, results, .csvs, and .hdf5s, appear in `./Trained Models/`

## Installation

We ran everything using a singularity container built from from Sing_Torch_05032024.def

## Setting up Datasets

Code-15 (100GB RAM, ~2 hours)
1. Download data from       https://zenodo.org/records/4916206
1. Place all .hdf5 in       `./RAW_DATA/Code15/TRAIN/exams_part0…17.hdf5`
1. Place exams.csv in       `./RAW_DATA/Code15/exams.csv`
1. Run                      `./RAW_DATA/Code15_ECG_Process_8020.py`          with `os.getcwd()` as ./ 
1. Move                     `./RAW_DATA/Code15_Processed/…`                  to     `./HDF5_DATA/Code15`

MIMIC-IV (300GB RAM, ~8 hours)
1. From “hosp” in  "https://physionet.org/content/mimiciv/2.2/  
	1. Download `/patients.csv.gz`         to `./RAW_DATA/MIMIC IV/`
	1. Download `/admissions.csv.gz`       to `./RAW_DATA/MIMIC IV/`

1. From https://physionet.org/content/mimic-iv-ecg/1.0/ 
	1. Download `machine_measurements.csv`                   to `./RAW_DATA/MIMIC IV/`
	1. Download `machine_measurements_data_dictionary.csv`   to `./RAW_DATA/MIMIC IV/`
	1. Download `record_list.csv`                            to `./RAW_DATA/MIMIC IV/`
	1. Download `waveform_note-links.csv`                    to `./RAW_DATA/MIMIC IV/`
	1. Download “files”                                    to `./RAW_DATA/MIMIC IV/ECGs/p1000….`
1. Run 				                            `./RAW_DATA/MIMIC_IV_PreProcess.py`
1. Run                                                        `./RAW_DATA/MIMIC_IV_Process.py`          with `os.getcwd()` as `./RAW_DATA`
1. Move                                                        `./RAW_DATA/MIMICIV_Processed/…`          to `./HDF5_DATA/MIMICIV`



## Use Overview:

Classifier-based survival models are built and evaluated through `Model_Runner_SurvClass.py`

PyCox survival models are built and evaluated through `Model_Runner_PyCox.py`

To manually build/evaluate models, use `Model_Runner_Caller.py`, which is pre-set with minimal calls to train a model.

To do this with job files, structure the args to Model_Runner_SurvClass/PyCox similarly to Model_Runner_Caller does it (more on this later).

Once you have several trained models, you can summarize them to tables using `Summarize_Trained_Models.py`
This creates the Summary_Tables folder, which includes: 
1. Trained_Model_Summary.csv (containing all training arguments, default arguments, and model metrics) 
1. Trained_Model_Main_Results_Table.csv (AUROC / AUPRC / all-time Brier and Concordance, averaged over all random seeds) 
1. Trained_Model_BrierConcordance_Table.csv (Brier and Concordance measures, averaged over all random seeds, right-censored to several time limits) 
1. Trained_Model_BS_BrierConcordance_Table.csv (Brier and Concordance measures, averaged over all bootstraps of all random seeds, right-censored ot several time limits)
1. Trained_Model_CrossvalTable.csv (1-yr AUROC / AUPRC, all-time Brier and Concordance, averaged over all random seeds)

Kaplan-Meier and histogram Survival Functions can be made by adapting Gen_KM_Curves.py to run on a particular training folder

We include Job_File_Maker_SurvClass and Job_File_Maker_PyCox - these generate job files in our institution's format, but can be adapted to yours.
To generate job files based on trained models, use Job_File_Maker_From_Summary.py after summarizing trained models with Summarize_Trained_Models.py
Job_File_Maker_From_Summary.py can be adapted to add / remove / change model runner arguments (ex: switch the evaluation dataset)
In our paper, models were trained with random seeds 10-14. Data was split with random seed 12345.

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

Patient ID is currently assumed to be in `Data['y_train'][:,0]` and `Data['y_test'][:,0]`. This is currently hard-coded but can be easily modified to a passed arg
We typically put ECG_ID in [:,1] (this doesn't matter), age in [:,2] (again, doesn't matter), 'time_to_event' in [:,3], and 'event' in [:,4]. time_to_event and event columns are passed args for both train and test.

We tried to make data processing flexible: inputs in the N-H (single-channel time series) or N-H-W (multi-channel time series) format are expanded to the image N-C-H-W format. Individual modelsl then change the shape as needed (an image model stays in N-C-H-W, ECG models tend to prefer N-H-W).


## See Also

[References here]

