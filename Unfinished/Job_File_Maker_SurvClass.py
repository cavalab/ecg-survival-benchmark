# -*- coding: utf-8 -*-
# This generates a bunch of job files to /Jobs_Generated/
# This might be specific to our institution, but can be adapted to yours.


"""
Created on Wed Nov  8 14:40:00 2023

@author: pxl28

# Note: use --partition=bch-gpu with --gres=gpu:Tesla_K:1 or --gres=gpu:Tesla_T:1 
# Note: use --partition=bch-gpu-pe with --gres=gpu:Titan_RTX:1 or --gres=gpu:Quadro_RTX:1 or --gres=gpu:NVIDIA_A40:1

#!/bin/bash
#SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:Titan_RTX:1

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email>

#SBATCH --time=00:05:00
#SBATCH --job-name=test6
#SBATCH --output=test6.txt

#SBATCH --ntasks=1
# #SBATCH --nodes=1
#SBATCH --mem=40G

module load singularity
singularity exec --nv hp5_cifar10.sif python3 'Train Model from Folder.py' --Train_Folder PROCESSED_DATA_1 --Model_Name Resnet18_0 --Train True --Test_Folder PROCESSED_DATA_1 --batch_size 128 --epoch_end 3
"""

import collections
collections.Callable = collections.abc.Callable

import numpy as np
import os
import io

# %% Function to glue strings together
def String_List_Append(Str1, Str2):
    return Str1 + [Str2]

# %% Function to build the non-arg part of the job file
def Get_Global_String_List( time_h, time_m, GPU, mem, model_type, name_suffix_list = ['091824','NoCov']):
    # returns a list of strings for a partial job file corresponding to inputs.
    # time_h - int
    # time_m - int
    # GPU - one of: Titan_RTX, Quadro_RTX, NVIDIA_A40, NVIDIA_A100, NVIDIA_L40, Tesla_K, Tesla_T, Any (<- budget for quadro)
    # model_type: 'Resnet18RegFlip', 'InceptionTimeReg', 'RibeiroReg'
    # name_chunks_list: list of details on the model to add to the name
    
    # assemble a full model name
    Model_Rand_ID = np.random.randint(1,1e7)
    name_suffix = [model_type] + name_suffix_list + [str(Model_Rand_ID)]
    Full_Model_Name = '_'.join(name_suffix)
    
    job_name = 'Run_' +  Full_Model_Name
    
    # begin assembling the job file
    String_List = []
    String_List = String_List_Append(String_List, '#!/bin/bash')
    
    #GPU
    if (GPU == 'Titan_RTX'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu-pe')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:Titan_RTX:1')
    elif (GPU == 'Quadro_RTX'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu-pe')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:Quadro_RTX:1')
    elif (GPU == 'NVIDIA_A40'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu-pe')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:NVIDIA_A40:1')
    elif (GPU == 'NVIDIA_L40'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu-pe')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:NVIDIA_L40:1')
    elif (GPU == 'NVIDIA_A100'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu-pe')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:NVIDIA_A100:1')
    elif (GPU == 'Tesla_K'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:Tesla_K:1')
    elif (GPU == 'Tesla_T'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:Tesla_T:1')
    elif (GPU == 'Any'):
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-gpu-pe')
        String_List = String_List_Append(String_List, '#SBATCH --gres=gpu:1')
    else:
        String_List = String_List_Append(String_List, '#SBATCH --partition=bch-compute') # the non-GPU
        
    String_List = String_List_Append(String_List, '#SBATCH --time='+"{:02d}".format(time_h)+':'+"{:02d}".format(time_m)+':00')
    String_List = String_List_Append(String_List, '#SBATCH --job-name='+job_name)
    String_List = String_List_Append(String_List, '#SBATCH --output=./Job_File_Out/Out_'+job_name+'.txt')
    String_List = String_List_Append(String_List, '#SBATCH --ntasks=1')
    String_List = String_List_Append(String_List, '#SBATCH --mem='+str(mem)+'G')
    
    String_List = String_List_Append(String_List, 'hostname') #  ... recommended for debugging
    
    # String_List = String_List_Append(String_List, 'module load singularity') # not needed since 5/1/24
    
    # And now we build the call
    Sing_Cmd = 'singularity exec --bind /lab-share --nv Sing_Torch_05032024.sif python3 \'Model_Runner_SurvClass.py\''
    
    Sing_Cmd = Sing_Cmd + ' ' + '--Model_Name ' + Full_Model_Name
    
    # Now we have a partial job file
    return String_List, Sing_Cmd, job_name
    
   
# %% Build many job files
# Okay. So we args_list. Each entry in that list is a list of strings.
# We want to create one job file for each combination of args in the args list
def decompress_args_list(args_list, Model_Type, Running_Arg = ''):
    # we're going to call this function recurisvely until the arg options have been split up into job files

    # no more args -> make job file
    if (len(args_list) == 0):
        make_single_job_file(Running_Arg, Model_Type)
        
    # in next arg, only one option -> append to String_List, then call self with one less arg
    elif (len(args_list[0]) == 1):
        if (len(args_list) == 1):
            decompress_args_list([], Model_Type, Running_Arg + args_list[0][0])
        else:
            decompress_args_list(args_list[1:], Model_Type, Running_Arg + args_list[0][0])
        
    # in next arg, multiple options, run self with each
    else:
        for k in args_list[0]:
            if (len(args_list) == 1):
                decompress_args_list([], Model_Type, Running_Arg + k)
            else:
                decompress_args_list(args_list[1:], Model_Type, Running_Arg + k)
    

def make_single_job_file(Running_Arg, Model_Type):    
    # By the time is is called we only have 'Running_Arg', which is a really long string of args we want to pass.
    
    # build a partial job file             
    String_List, Sing_Cmd, job_name = Get_Global_String_List(time_h,time_m, GPU, mem, Model_Type)
    
    # Add the args
    String_List = String_List_Append(String_List, Sing_Cmd + Running_Arg)
    print(String_List)
    
    # Save it out in a unix-compatible manner.
    # https://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python-with-newlines
    # https://stackoverflow.com/questions/2536545/how-to-write-unix-end-of-line-characters-in-windows
    
    tmp = os.path.join(os.getcwd(), 'Jobs_Generated')
    if (os.path.isdir(tmp) == False):
        os.mkdir(tmp)
    targ_dir = os.path.join(os.getcwd(), 'Jobs_Generated','SurvClass')
    if (os.path.isdir(targ_dir) == False):
        os.mkdir(targ_dir)
    file_path = os.path.join(targ_dir, job_name+'.txt')
    
    with io.open(file_path,'w', newline='\n') as f:
        for line in String_List:
            f.write(f"{line}\n")
            
# %% Set Global Params
time_h = 48 # hour # BCH - 6, Code15 - 15. Double for 'Any' GPU
time_m = 00 # min

GPU = 'Any' # Quadro_RTX or Titan_RTX or Tesla_K or Tesla_T or NVIDIA_A40 or or NVIDIA_A100 'Any' (Any is good for Eval) 
mem = 99 # GB, must be int. MIMICIV has taken at most 249GB so far, Code15 99GB

# %% Set Sweep params



# running: KM and KW sweep for ribeiro and inceptiontime assuming adamw
# next: fix ribeiro and inceptiontime KM and KW values, run adam v sched v cocob 6x wd levels


# each element of args_list MUST begin with ' --' (including the space)

Model_Type_List = ['Ribeiro', 'InceptionTime'] # Ribeiro, InceptionTime
for Model_Type in Model_Type_List:
    glob_args_list = [] # args_list is a list of lists. if an appended list has more than one entry, generate job files per entry

    folders = ['Code15'] # ['BCH_ECG', 'Code15', 'MIMICIV']
    glob_args_list.append([ ' --Test_Folder ' + k + ' --Train_Folder ' + k for k in folders ])
        
    horizons = [1,2,5,10]
    glob_args_list.append([ ' --horizon ' + str(float(k)) for k in horizons ])
    
    glob_args_list.append([ ' --Eval_Dataloader Test'  ]) # 'Validation' or 'Train' (will be shuffled). defaults to test    
    glob_args_list.append([ ' --Rand_Seed '+ str(k) for k in [10,11,12,13,14]  ])  
    
    
    glob_args_list.append([  ' --y_col_train_time 3'  ]) # code15; bch; MIMICIV - time 3, event 4
    glob_args_list.append([ ' --y_col_train_event 4'  ]) 
    glob_args_list.append([   ' --y_col_test_time 3'  ])
    glob_args_list.append([  ' --y_col_test_event 4'  ])
    # print('job file time/event in MIMICIV format!')
    
    glob_args_list.append([ ' --Train True'  ])
    # args_list.append([ ' --Load Best'  ]) # exclusive with train
    
    glob_args_list.append([ ' --epoch_end 200'  ])
    
    Early_Stops = ['20'] #Survival Models: 'LH', 'DeepHit', 'MTLR', 'CoxPH'
    glob_args_list.append([ ' --early_stop ' + k for k in Early_Stops ])
    
    glob_args_list.append([ ' --Validate_Every 1'  ])
    
    glob_args_list.append([ ' --batch_size 512'  ])
    glob_args_list.append([ ' --GPU_minibatch_limit 64'  ])  # how many to run on GPU at a time. divisor of batch_size. 256 for CNNs, 64 for Spect, 32? for LSTM 
    
    glob_args_list.append([ ' --Norm_Func nchW'  ])          # nchW, nChw, or None

    Optimizer = 'Adam'                                  # 'cocob' or 'Adam'
    glob_args_list.append([ ' --optimizer ' + Optimizer  ])         
    glob_args_list.append([ ' --Scheduler True'  ])
    
    glob_args_list.append([ ' --Loss_Type CrossEntropyLoss'  ])
    
    # Cov_Arg_List = ['[1,2]'] # MIMIC
    # Cov_Arg_List = ['[2,5]'] # Code-15
    # glob_args_list.append([ ' --val_covariate_col_list ' + k + ' --test_covariate_col_list ' + k for k in Cov_Arg_List ])
    
    glob_args_list.append([ ' --fusion_layers 3'  ])
    glob_args_list.append([ ' --fusion_dim 128'  ])
    
    glob_args_list.append([ ' --cov_layers 3'  ])
    glob_args_list.append([ ' --cov_dim 32'  ])
    
    
    
    # Optimizer = 'cocob'
    # if Optimizer == 'cocob':
    #     wds = ['0.3','0.1','0.03','0.01','0.003','0.001'] # weight decays for cocob
    #     glob_args_list.append([ ' --cocob_wd ' + k for k in wds ])
    #     glob_args_list.append([ ' --optimizer ' + Optimizer  ]) 
    # glob_args_list.append([ ' --Scheduler False'  ])

    # prep
    decompress_args_list(glob_args_list, Model_Type)
    
 
                                        