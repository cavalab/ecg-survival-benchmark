# -*- coding: utf-8 -*-
"""
This file generates a set of arguments, just as in a command-line call, 
and issues them to Model_Runner.

Run one section at a time - this is built around debugging
"""

# %% Classifier-Cox
import collections
collections.Callable = collections.abc.Callable

import Model_Runner_Classifier_Cox

args  = '--Train_Folder MIMIC_DEBUG'         # Training Data Folder Name
args += ' --Model_Name ZeroNet_Classif_3'    # Must start with ModelType. InceptionTime or Ribeiro (ResNet) or ZeroNet (Returns a 0)
# args += ' --Load Best'                     # if you want to load a model you've trained before - largely obsolete and can skip, since latest model is loaded when you request training
args += ' --Test_Folder MIMIC_DEBUG'         # Test Data Folder Name
args += ' --batch_size 512'                  # batch size
args += ' --epoch_end 1'                     # set to -1 if you don't want to train the model
args += ' --validate_every 1'                # default; how often to get validation performance
args += ' --Save_Out_Checkpoint True'        # default; are you saving out the checkpoint?
args += ' --GPU_minibatch_limit 32'          # how many samples go to GPU at one time (depends on GPU)
args += ' --Eval_Dataloader Test'            # default; 'Test' or 'Train' or 'Validate' - what you want performance measures on
args += ' --optimizer Adam'                  # default; only option
args += ' --Scheduler True'                  # default; only option
args += ' --Loss_Type CrossEntropyLoss'      # default; not only option
args += ' --early_stop 25'                   # stop after this many epochs with no improvement
args += ' --y_col_train_time 3'              # 
args += ' --y_col_train_event 4'             #
args += ' --y_col_test_time 3'               # 
args += ' --y_col_test_event 4'              # 
args += ' --Rand_Seed 10'                    # Random seed. randomly generated if not specified
args += ' --horizon 0.0822'                  # What time (years) classifier should optimize for 
args += ' --Norm_Func nchW'                  # Normalize ECG per channel based on the training data set (requires dataset load since Tr/Va split is random-seed dependent)

args += ' --val_covariate_col_list [1,2]'    # Which training covariates to include
args += ' --test_covariate_col_list [1,2]'   # Which test set covariates to include

args += ' --fusion_layers 3'                 # Fusion model details
args += ' --fusion_dim 128'
args += ' --cov_layers 3'
args += ' --cov_dim 32'

args += ' --debug False'                    # 'True' limits data to 1k samples of tr/val/test. good for debug.

args += ' --Multimodal_Out True'            # Saves out final model details - PID, TTE, E, Prediction, Correct_Prediction
args += ' --Neg_One_Out True'               # Saves out model features


args = args.split(' ')
Model_Runner_Classifier_Cox.Run_Model_via_String_Arr(args)

# %% Deep Vurvival - PyCox - models. 
import collections
collections.Callable = collections.abc.Callable

import Model_Runner_Deep_Survival

args  = '--Train_Folder MIMIC_DEBUG'         # Training Data Folder Name
args += ' --Model_Name ZeroNet_PYCox_3'      # Must start with ModelType. InceptionTime or Ribeiro (ResNet) or ZeroNet (Returns a 0)
args += ' --Test_Folder MIMIC_DEBUG'         # Test Data Folder Name
args += ' --batch_size 512'                  # batch size
args += ' --epoch_end 5'                     # set to -1 if you don't want to train the model after initialization and loading
args += ' --validate_every 1'                # default; how often to get validation performance
args += ' --Save_Out_Checkpoint True'        # default; are you saving out the checkpoint?
args += ' --GPU_minibatch_limit 32'          # how many samples go to GPU at one time?
args += ' --Eval_Dataloader Test'            # default; 'Test' or 'Train' or 'Validate' - what you want performacne measures on
args += ' --optimizer Adam'                  # should be default and only option
args += ' --Scheduler True'                  # should be default, but is not only option
args += ' --early_stop 25'                   # Stop after this many epochs with no improvement
args += ' --y_col_train_time 3'              #
args += ' --y_col_train_event 4'             #
args += ' --y_col_test_time 3'               # 
args += ' --y_col_test_event 4'              # 
args += ' --Rand_Seed 10'                    # Random seed. randomly generated if not specified

args += ' --pycox_mdl LH'                    # The deep survival model: CoxPH, LH, MTLR, DeepHit
args += ' --Norm_Func nchW'                  # Normalize ECG per channel based on the training data set

args += ' --val_covariate_col_list [1,2]'    # Which training covariates to include
args += ' --test_covariate_col_list [1,2]'   # Which test set covariates to include

args += ' --fusion_layers 3'                 # Fusion model details
args += ' --fusion_dim 128'
args += ' --cov_layers 3'
args += ' --cov_dim 32'

args += ' --debug True'                      # 'True' limits data to 1k samples of tr/val/test.

args += ' --Multimodal_Out True'             # Saves out final model details - PID, TTE, E, Prediction, Correct_Prediction
args += ' --Neg_One_Out True'                # Saves out model features


args = args.split(' ')
Model_Runner_Deep_Survival.Run_Model_via_String_Arr(args)


# %% XGB (into Cox)
import collections
collections.Callable = collections.abc.Callable

import Model_Runner_XGB

args  = '--Train_Folder MIMIC_DEBUG'         # Training Data Folder Name
args += ' --Model_Name XGB_Test'             # Must start with ModelType. InceptionTime or Ribeiro (ResNet) or ZeroNet (Returns a 0)
args += ' --Test_Folder MIMIC_DEBUG'         # Test Data Folder Name

args += ' --y_col_train_time 3'              #
args += ' --y_col_train_event 4'             #
args += ' --y_col_test_time 3'               # 
args += ' --y_col_test_event 4'              # 

args += ' --Rand_Seed 10'                    # Random seed. randomly generated if not specified
args += ' --horizon 1'                       # What time (years) classifier should optimize for 

args += ' --val_covariate_col_list [1,2,6,8,10,12,14,16,18,20]' # MIMIC - age/sex and machine measures
args += ' --test_covariate_col_list [1,2,6,8,10,12,14,16,18,20]'

args += ' --debug False'                     # 'True' limits data to 1k samples of tr/val/test.
args += ' --provide_data_details True'       # prints out 'Table 1' details

args = args.split(' ')
Model_Runner_XGB.Run_Model_via_String_Arr(args)