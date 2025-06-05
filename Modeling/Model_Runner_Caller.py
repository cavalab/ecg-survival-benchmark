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

args  = '--Train_Folder MIMICIV'         # Training Data Folder Name
args += ' --Model_Name InceptionTime_CC_Debug'    # Must start with ModelType. InceptionTime or Ribeiro (ResNet) or ZeroNet (Returns a 0)
# args += ' --Load Best'                     # if you want to load a model you've trained before - largely obsolete and can skip, since latest model is loaded when you request training
args += ' --Test_Folder MIMICIV'         # Test Data Folder Name
args += ' --batch_size 512'                  # batch size
args += ' --epoch_end 25'                     # set to -1 if you don't want to train the model
args += ' --validate_every 1'                # default; how often to get validation performance
args += ' --Save_Out_Checkpoint True'        # default; are you saving out the checkpoint?
args += ' --GPU_minibatch_limit 128'          # how many samples go to GPU at one time (depends on GPU)
args += ' --Eval_Dataloader Test'            # default; 'Test' or 'Train' or 'Validate' - what you want performance measures on
args += ' --optimizer Adam'                  # default; only option
args += ' --Scheduler True'                  # default; only option
args += ' --Loss_Type CrossEntropyLoss'      # default; not only option
args += ' --early_stop 25'                   # stop after this many epochs with no improvement
args += ' --Rand_Seed 10'                    # Random seed. randomly generated if not specified
args += ' --horizon 2'                  # What time (years) classifier should optimize for 
args += ' --Norm_Func nchW'                  # Normalize ECG per channel based on the training data set (requires dataset load since Tr/Va split is random-seed dependent)

# args += ' --covariates [Age,Is_Male]'        # Which training covariates to include
args += ' --covariates [Age,Is_Male,P_Axis,QRS_Axis,T_Axis,RR,P_Dur,PQ_Dur,QRS_Dur,QT_Dur]'

args += ' --fusion_layers 3'                 # Fusion model details
args += ' --fusion_dim 512'
args += ' --cov_layers 3'
args += ' --cov_dim 32'

args += ' --debug True'                    # 'True' limits data to 5k samples of tr/test. At loading time.

# args += ' --Multimodal_Out True'            # Saves out final model details - PID, TTE, E, Prediction, Correct_Prediction
# args += ' --Neg_One_Out True'               # Saves out model features


args = args.split(' ')
Model_Runner_Classifier_Cox.Run_Model_via_String_Arr(args)

# %% Deep Survival - PyCox - models. 
import collections
collections.Callable = collections.abc.Callable

import Model_Runner_Deep_Survival

args  = '--Train_Folder MIMICIV'         # Training Data Folder Name
args += ' --Model_Name InceptionTime_042525_Cov_MIMICIV_303463'   # Must start with ModelType. InceptionTime or Ribeiro (ResNet) or ZeroNet (Returns a 0) or ECGTransForm
args += ' --Test_Folder MIMICIV'         # Test Data Folder Name
args += ' --batch_size 512'                  # batch size
args += ' --epoch_end -1'                     # set to -1 if you don't want to train the model after initialization and loading
args += ' --validate_every 1'                # default; how often to get validation performance
# args += ' --Save_Out_Checkpoint True'        # default; are you saving out the checkpoint?
args += ' --GPU_minibatch_limit 64'          # how many samples go to GPU at one time?
args += ' --Eval_Dataloader Test'            # default; 'Test' or 'Train' or 'Validate' - what you want performacne measures on
args += ' --optimizer Adam'                  # should be default and only option
args += ' --Scheduler True'                  # should be default, but is not only option
args += ' --early_stop 100'                   # Stop after this many epochs with no improvement

args += ' --Rand_Seed 13'                    # Random seed. randomly generated if not specified

args += ' --pycox_mdl MTLR'                    # The deep survival model: CoxPH, LH, MTLR, DeepHit
args += ' --Norm_Func nchW'                  # Normalize ECG per channel based on the training data set

args += ' --covariates [Age,Is_Male]'        # Which training covariates to include
# args += ' --covariates [Age,Is_Male,P_Axis,QRS_Axis,T_Axis,RR,P_Dur,PQ_Dur,QRS_Dur,QT_Dur]'

args += ' --fusion_layers 3'                 # Fusion model details
args += ' --fusion_dim 128'
args += ' --cov_layers 3'
args += ' --cov_dim 32'

args += ' --debug True'                      # 'True' limits data to 1k samples of tr/val/test.

args += ' --Multimodal_Out False'             # Saves out final model details - PID, TTE, E, Prediction, Correct_Prediction
args += ' --Neg_One_Out False'                # Saves out model features


args = args.split(' ')
Model_Runner_Deep_Survival.Run_Model_via_String_Arr(args)


# %% XGB (into Cox)
import collections
collections.Callable = collections.abc.Callable

import Model_Runner_XGB

args  = '--Train_Folder MIMICIV'         # Training Data Folder Name
args += ' --Model_Name XGB_Test3'             # Must start with ModelType. InceptionTime or Ribeiro (ResNet) or ZeroNet (Returns a 0)
args += ' --Test_Folder MIMICIV'         # Test Data Folder Name

args += ' --Rand_Seed 10'                    # Random seed. randomly generated if not specified
args += ' --horizon 1'                       # What time (years) classifier should optimize for 

# args += ' --covariates [Age,Is_Male]'        # Which training covariates to include
args += ' --covariates [Age,Is_Male,P_Axis,QRS_Axis,T_Axis,RR,P_Dur,PQ_Dur,QRS_Dur,QT_Dur]'

# args += ' --debug False'                     # 'True' limits data to 1k samples of tr/val/test.
# args += ' --provide_data_details True'       # prints out 'Table 1' details

args = args.split(' ')
Model_Runner_XGB.Run_Model_via_String_Arr(args)