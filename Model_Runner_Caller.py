# -*- coding: utf-8 -*-
"""
This file generates a set of arguments, just as in a command-line call, 
and issues them to Model_Runner

It's built for quick debugging

# Note: model names BEGIN with the type of model (survival: RibeiroReg or InepctionTimeReg), then an underscore, then an identifier ("Bob")
"""

# %% Survival models

import collections
collections.Callable = collections.abc.Callable
import Model_Runner_PyCox 

# Train a model
args = '--Train_Folder Code15 --Model_Name RibeiroReg_asdf2 --epoch_end 2 --early_stop 50 --Test_Folder Code15 --batch_size 512 --GPU_minibatch_limit 64 --y_col_train_time 3 --y_col_train_event 4 --y_col_test_time 3 --y_col_test_event 4 --Rand_Seed 3 --pycox_mdl LH --Eval_Dataloader Test --Scheduler True --Norm_Func nchW'

# Evaluate that same model without training (note "Load Best" and "epoch_end -1")
# args = '--Train_Folder Code15 --Model_Name RibeiroReg_asdf2 --Load Best --epoch_end -1 --early_stop 50 --Test_Folder Code15 --batch_size 512 --GPU_minibatch_limit 64 --y_col_train_time 3 --y_col_train_event 4 --y_col_test_time 3 --y_col_test_event 4 --Rand_Seed 3 --pycox_mdl LH --Eval_Dataloader Test --Scheduler True --Norm_Func nchW'

args = args.split(' ')
Model_Runner_PyCox.Run_Model_via_String_Arr(args)

# %% Classifiers and Regressions. (same args hold for Survival models)
import collections
collections.Callable = collections.abc.Callable

import Model_Runner_SurvClass

args  = '--Train_Folder Code15'
args += ' --Model_Name InceptionClass_asdf2' #InceptionClass or RibeiroClass
# args += ' --Load Best'                     # if you want to load a model you've trained before
args += ' --Test_Folder Code15'
args += ' --batch_size 512'
args += ' --epoch_end 5'                     # set to -1 if you don't want to train the model after initialization and loading
args += ' --validate_every 1'
args += ' --Save_Out_Checkpoint True'
args += ' --GPU_minibatch_limit 32'          # how many samples go to GPU at one time
args += ' --Eval_Dataloader Test'            # 'Test' or 'Train' or 'Validate' - what you want performacne measures on
args += ' --optimizer Adam'                  # should be default and only option
args += ' --Scheduler True'                  # should be default, but is not only option
# args += ' --debug True'                    # 'True' limits data to 1k samples of tr/val/test.
args += ' --Loss_Type CrossEntropyLoss' 
args += ' --early_stop 25'                   # Stop after this many epochs with no improvement
args += ' --y_col_train_time 3'              #
args += ' --y_col_train_event 4'             #
args += ' --y_col_test_time 3'               # <- these depend on the data format
args += ' --y_col_test_event 4'              # 
args += ' --Rand_Seed 13'                    # Random seed
args += ' --horizon 10.0'                    # What time (years) models should try to optimize for [Classifier specific]
args += ' --Norm_Func nchW'                  # Normalize ECG per channel based on the training data set


args = args.split(' ')
Model_Runner_SurvClass.Run_Model_via_String_Arr(args)
