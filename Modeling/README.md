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

Classifier-based survival models are built and evaluated through `Model_Runner_Classifier_Cox.py`

PyCox survival models are built and evaluated through `Model_Runner_Deep_Survival.py`

To manually build/evaluate models, use `Model_Runner_Caller.py`, which is pre-set with minimal calls to train a model.

To do this with job files, structure the args to Model_Runner_Classifier_Cox/Deep_Survival similarly to how Model_Runner_Caller does it (more on this later).

Once you have several trained models, you can summarize them to tables using `Analysis_Summarize_Trained_Models.py`
This creates the Summary_Tables folder, which includes: 
1. Trained_Model_Summary.csv (containing all training arguments, default arguments, and model metrics) 
1. Trained_Model_Main_Results_Table.csv (AUROC / AUPRC / all-time Brier and Concordance, averaged over all random seeds) 
1. Trained_Model_BrierConcordance_Table.csv (Brier and Concordance measures, averaged over all random seeds, right-censored to several time limits) 
1. Trained_Model_BS_BrierConcordance_Table.csv (Brier and Concordance measures, averaged over all bootstraps of all random seeds, right-censored ot several time limits)
1. Trained_Model_CrossvalTable.csv (1-yr AUROC / AUPRC, all-time Brier and Concordance, averaged over all random seeds)

Kaplan-Meier and histogram Survival Functions can be made by adapting Gen_KM_Curves.py to run on a particular training folder

We include Job_File_Maker_Classifier_Cox and Job_File_Maker_Deep_Survival - these generate job files in our institution's format, but can be adapted to yours.
To generate job files based on trained models, use Job_File_Maker_From_Summary.py after summarizing trained models with Summarize_Trained_Models.py
Job_File_Maker_From_Summary.py can be adapted to add / remove / change model runner arguments (ex: switch the evaluation dataset)
In our paper, Test/Train Data was split with random seed 12345. Training data was then further split in to Train/Validation with random seeds 10-14. 

## Code Structure:

Model_Runner_* loads data and splits it into ECG Data['x_train'], Data['x_test'], Data['x_valid'], and Label(y) and Covariate(z) equivalents. Then it more-or-less calls model.init(), model.load(), model.fit(), and model.eval(), then sets up output folders, evaluates the models, and saves plots, .csvs, and .hdf5s with results. All data is stored to RAM and moved around with the Data dictionary. (It would not be too difficult to adjust this to .hdf5 sample-by-sample loading to reduce RAM). Model_Runner_Support contains shared scripts.

When Model_Runner* is instructed to train a model (Epoch_End > 0), it will check to see if a model checkpoint already exists to continue training from that checkpoint. This loads the CUDA and CPU random seeds so proceeds deterministically. Training can be skipped by adding 'load' arguments ('--Load Best' or '--Load Last') and setting '--Epoch_End -1'

The models are instances of GenericModelClassifierCox/DeepSurvival, which are themselves objects of class GenericModel. More generic functions (like wrapping an ECG-processing architecture with a 'fusion' module) are located in GenericModel, but more specific functions (like training PyCox models)would be in GenericModelDeepSurvival. Support_Functions contains additional shared scripts. Model_Runner_XGB is much more lightweight, and also prints data demographics (combining test/train).

Passing Arguments: Everything is interpreted via a series of string-string argument-value pairs. Arguments must begin with "--" and be separated by spaces. Model_Runner_* converts these to a string:string dictionary, 'args', mapping arguments to values that is passed throughout the code. If you want a new architecture to take a parameter as an argument, you can include it in the call and it will be accessible during architecture initialization.

Trained models are stored in /Trained_Models, which separates models by training data source. Beyond that, models are identified by their name, which begins with the architecture (e.g. InceptionTime_1). Model evaluations are further stored in /Eval and also separated by test data source. e.g. /Trained_Models/InceptionTime_1/Eval/Code-15/ 

A GPU is required for both model training and evaluation. We do save the model outputs, however, so metrics can be computed off-cluster, as we did for subgroup analysis. 

Importantly this also means that /Trained_Models/ contains copies of the test data

## Example Job File:

See Job_File_Maker_PyCox.py 

## On Data formatting:

Patient ID is currently assumed to be in `Data['y_train'][:,0]` and `Data['y_test'][:,0]`. This is currently hard-coded but can be easily modified to a passed arg
We typically put ECG_ID in [:,1] (this doesn't matter), age in [:,2] (again, doesn't matter), 'time_to_event' in [:,3], and 'event' in [:,4]. time_to_event and event columns are passed args for both train and test.

We tried to make data processing flexible: inputs in the N-H (single-channel time series) or N-H-W (multi-channel time series) format are expanded to the image N-C-H-W format. Individual modelsl then change the shape as needed (an image model stays in N-C-H-W, ECG models tend to prefer N-H-W).


## See Also

