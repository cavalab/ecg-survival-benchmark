# args['horizon']           - float; {coreq} also requires args['y_col_train_time'] and args['y_col_train_event'], or args['y_col_test_time'] and args['y_col_test_event']. ignores args['y_col_train'] and args['y_col_test']
# args['y_col_train_time']  - int;   {coreq 'horizon'} 
# args['y_col_train_event'] - int;   {coreq^}
# args['y_col_test_time']   - int;   {coreq 'horizon'}
# args['y_col_test_event']  - int;   {coreq^}
# args['y_col_train']       - int; which column of Y is train data (ignored with horizon)
# args['y_col_test']        - int; which column of Y is test data  (ignored with horizon)

# New asusmption 5/1/24: column 0 of Data[] is patient ID

"""
MODEL_RUNNER

Model_Runner:
    - Model_Runner Interprets passed arguments
        - Either from Command Line (main)
        - Or from another script (Run_Model_via_String_Arr)
            ex: args = '--Train_Folder '+folder+' --Model_Name Resnet18Class_asfd1 --Train True --Test_Folder '+folder+' --batch_size 128 --epoch_end 10 --validate_every 10'

    - Quick note on Models:
        - Model Type is set by the first half (up to an underscore) of their name: --Model_Name Resnet18Class_[asfd1]
        - Trained models live in Trained_Models/[their training data folder], as passed in: --Train_Folder [folder]
        - Model_Runner MUST get the Train_Folder and Model_Name arguments to know which model to build or use
        
    - Model_Runner:
        - Sets Model Name 
        - Determines if training happens (requires --Train True AND --Train_Folder [folder])
        - Sets a random seed if that was not passed
        - If training, Loads training data. 
            - Data is currently (12/4/23) assumed to be in:
                - In os.path.join(os.getcwd(),'HDF5_DATA',args['Train_Folder'],'TRAIN_DATA','Train_Data.hdf5')
                - With numpy parameters 'x' and 'y'
        - If evaluating: (requires --Test_Folder [folder])
            - Attempts to load from os.path.join(os.getcwd(),'HDF5_DATA',args['Test_Folder'],'TEST_DATA','Test_Data.hdf5')
            - On failure:
                - Checks for a passed TE_RATIO argument. By default, TR_RATIO = 80, VA_RATIO = 20.
                    - generates a test set from a training/validation/test split of:
                    - os.path.join(os.getcwd(),args['Test_Folder'],'TRAIN_DATA','Train_Data.hdf5')
        - If a training set is present:
            - Data is split randomly into Train/Test/Validate from TR_RATIO/TE_RATIO/VA_RATIO
            - Splits prioritize allocations for Train/Test/Validate in that order (ceil, ceil, remainder) (If a class only has one sample, it is in training.)
            - For classifiers, the split is performed per class
            - Everything is stored in Data['x_train'], Data['x_valid'], Data['x_test'], and y-equivalents
            
        - Initalizes a model: model(args, Data)
        - Runs model.load() if --Load [anything]
        - Runs model.eval() if --Test_Folder [anything]
        - Generates figures from eval, saves figure, eval params, and eval outputs
        
    - Other:
        - Model training and evaluation are split intentionally
        - Classifiers currently assume the lowest class number to be 0
        - If you specify a non-default (0 or 1) column of train_y or test_y to use, that will add sub-folders to the path
        

main() parses command line inputs, then feeds a dict to run_model()
Run_Model_via_String_Arr() parses arguments from an array of strings, then feeds a dict to run_model()

DATA is assumed to be in N-H or N-H-W or N-C-H-W format. 
For time series, this would be N-Length or N-Length-Chan or N-[Color]-Length-Chan
Currently, all models reshape it into N-C-H-W

[Not yet implemented] normalization is currently done per_color (N-_C_-H-W)
"""

# %% Imports; Support functions before main functions...

# handle pycox folder requirement FIRST
import os 
os.environ['PYCOX_DATA_DIR'] = os.path.join(os.getcwd(),'Mandatory_PyCox_Dir') # next line requires this. it's a bad init call.
from pycox.utils import kaplan_meier

from MODELS import Support_Functions
from MODELS import InceptionTimeClassifier
from MODELS import Ribeiro_Classifier
from MODELS import FFClass
from MODELS import LSTMClass
from MODELS import TimesNetClass

from MODELS.Support_Functions import get_surv_briercordance # only get useful time points (bootstraps)
from MODELS.Support_Functions import get_AUROC_AUPRC
from MODELS.Support_Functions import Save_to_hdf5


import pandas as pd
import matplotlib.pyplot as plt

from sksurv.linear_model import CoxPHSurvivalAnalysis
from scipy.special import softmax 

import h5py
import numpy as np
import torch
import time
import json

import collections
collections.Callable = collections.abc.Callable

import argparse 

# to see how the classifier did at its job
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# %% 
def main(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args()
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
def Run_Model_via_String_Arr(*args):
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Just convert the args to a string-string dict so each model handles its own parsing.
    _, unknown_args = parser.parse_known_args(args[0])
    args = dict(zip([k[2:] for k in unknown_args[:-1:2]],unknown_args[1::2])) # from stackoverflow
    print(args) #these are all strings!
    Run_Model(args)
    
    # 
def Run_Model(args):
    # input is paired dict of strings named args
    start_time = time.time()
    
    #%%0. Toggleable speedup for debug
    debug  = False
    if ('debug' in args.keys()):
        if args['debug'] == 'True':
            debug = True
            sub_len = 1000
            if (debug):
                print("Debug speedup! only using "+str(sub_len)+' elems of tr/val/test!')

    # %% 1. CUDA check
    # CUDA
    for i in range(torch.cuda.device_count()):
       print(torch.cuda.get_device_properties(i).name)
       
    if (torch.cuda.is_available() == False):
        print('No CUDA. Exiting.')
        exit()
       
    # %% 2. Arg Processing
    # Grab model name. No point in proceeding without it.
    if ('Model_Name' not in args.keys()):
        print('Model_Name not specified - cant train or pull models')
        exit()

    Model_Type = args['Model_Name'].split('_')[0]
    
    if ('Train_Folder' not in args.keys()):
        print('Train_Folder not specified - cant train or pull models')
        exit()
    
    # %% 3. Random Seeds should really be from args. Note: "load"ed models overwrite these!
    if ('Rand_Seed' in args.keys()):
        args['Rand_Seed'] = int(args['Rand_Seed'])
        
    if ('Rand_Seed' not in args.keys()):
        np.random.seed()
        args['Rand_Seed'] = np.random.randint(70000,80000)
        print('Rand Seed Not Set. Picking a random number 70,000 - 80,000... ' + str(args['Rand_Seed']))    
    
    
    np.random.seed(args['Rand_Seed'])
    torch.manual_seed(args['Rand_Seed'])
    torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
    
    # %% 4. Data load - NeurIPS simplified. Assume we are pulling from 'train_folder' and 'test_folder'
    Data = {}
    datapath1 = os.path.dirname(os.getcwd()) # cleverly jump one one folder without referencing \\ (windows) or '/' (E3)
    with h5py.File(os.path.join(datapath1,'HDF5_DATA',args['Train_Folder'],'TRAIN_DATA','Train_Data.hdf5'), "r") as f:
        Data['x_train'] = f['x'][()]
        Data['y_train'] = f['y'][()]
        Train_Col_Names = [k.decode('UTF-8') for k in f['column_names'][()]]
    
    with h5py.File(os.path.join(datapath1,'HDF5_DATA',args['Test_Folder'],'TEST_DATA','Test_Data.hdf5'), "r") as f:
        Data['x_test'] = f['x'][()]
        Data['y_test'] = f['y'][()] 
        Test_Col_Names = [k.decode('UTF-8') for k in f['column_names'][()]]
        
    print('Loaded Train and Test. Total time elapsed: ' + str(time.time()-start_time) )
        
                
    
    # %% 5 Data clean
    # Sometimes time-to-event is '-1.0' meaning no follow up time.
    # Sometimes TTE < 0 (recording error?)
    # Find and trash those indices    
    for key in ['y_train', 'y_test']:
        if ( (key in Data.keys()) and (len(Data[key].shape) > 1) ):
            x_key = 'x' + key[1:]
            
            # mark negative TTE
            neg_inds = np.where(Data[key][:,int(args['y_col_train_time'])] < 0)[0]
            inds_to_del = neg_inds.tolist()
            
            # mark nan traces (5x faster)
            for i in range(Data[x_key].shape[0]):
                if (np.isnan(Data[x_key][i]).any()):
                    if i not in inds_to_del:
                        inds_to_del.append(i)
            
            # mark nan traces (pre 8/14/24)
            # nan_inds = np.where(np.sum(np.sum(np.isnan(Data[x_key]),axis=2),axis=1) > 0)[0]
            # for k in nan_inds:
            #     if k not in inds_to_del:
            #         inds_to_del.append(k)
            
                     
            # remove data, avoid calling np.delete on Data cause that doubles RAM - just select the indices to keep instead
            if (len(inds_to_del) > 0):
                print('removing ' + str(len(inds_to_del)) + ' inds with nan or negative time')
                inds_to_keep = np.delete( np.arange(Data[key].shape[0]), inds_to_del )
                Data[x_key] = Data[x_key][inds_to_keep] 
                Data[key] = Data[key][inds_to_keep] 
            
    print('Checked data for negative time and nan ECG. Total time elapsed: ' + str(time.time()-start_time) )

    # %% 7. Horizoning
    # We want our model to determine whether an event occurs by time H, trained with a classifier loss function.
    # later, we'll build cox models on the output and evaluate those survival functions on the actual data 
    # needs args['y_col_train_time'],args['y_col_train_event'],args['y_col_test_time'],args['y_col_test_event']
    assert('horizon' in args.keys())
    assert('y_col_train_time' in args.keys())
    assert('y_col_train_event' in args.keys())
    assert('y_col_test_time' in args.keys())
    assert('y_col_test_event' in args.keys())
    
    # expand Data['y_train'] with extra columns showing PID, TTE*, and E*, where TTE* and E* are the times/events we are training on. Don't modify original TTE/E, which we evaluate survival models on.
    assert ( ('y_train' in Data.keys()) and (len(Data['y_train'].shape) > 1) )
    print('model_runner_survclass_seq: limiting training TTE to time horizon! assuming class 1 if event by H, else 0 (incl censor)')
    
    horizon = float(args['horizon'])
    times_below_h = Data['y_train'][:,int(args['y_col_train_time'])] <= horizon
    event = abs (Data['y_train'][:,int(args['y_col_train_event'])] -1) < 1e-4
    event_mod = (times_below_h*event).astype(int) # did event happen? Leave a (k,) shape array
    
    # expand y_train - append PID, TTE*, E* @ column inds [-3,-2,-1], respectively
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,0],1)), axis=1)
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(Data['y_train'][:,int(args['y_col_train_time'])],1)), axis=1)
    Data['y_train'] = np.concatenate((Data['y_train'], np.expand_dims(event_mod,1)), axis=1)

    # Do the same for Data['y_test]
    assert ( ('y_test' in Data.keys()) and (len(Data['y_test'].shape) > 1) )
    print('model_runner_survclass_seq: limiting test TTE to time horizon! assuming class 1 if event by H, else 0 (incl censor)')

    horizon = float(args['horizon'])
    times_below_h = Data['y_test'][:,int(args['y_col_test_time'])] <= horizon
    event = abs (Data['y_test'][:,int(args['y_col_test_event'])] -1) < 1e-4
    event_mod = (times_below_h*event).astype(int) # did event happen? Leave a (k,) shape array
    
    # expand y_test - append PID, TTE*, E* @ column inds [-3,-2,-1], respectively
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,0],1)), axis=1)
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(Data['y_test'][:,int(args['y_col_test_time'])],1)), axis=1)
    Data['y_test'] = np.concatenate((Data['y_test'], np.expand_dims(event_mod,1)), axis=1)

    print('Computed event marker for horizon. Total time elapsed: ' + str(time.time()-start_time) )

    # %% 8. Split loaded "training" data RANDOMLY BY PATIENT ID
    if ('y_train' in Data.keys()):
        
        # NeurIPS version - split Training dataset 80 / 20 into Tr/Val
        TR = 80
        VA = 20
        TE = 00
        
        # Per ID, find matching data rows      
        Subj_IDs = Data['y_train'][:,-3]        # PID is now at -3    
        Subj_IDs_Unique = np.unique(Subj_IDs)
        
        #Speedup 08/14/24
        Subj_ID_to_Rows_Dict = {} # map ID to rows
        for ind,val in enumerate(Subj_IDs):
            if val in Subj_ID_to_Rows_Dict.keys():
                Subj_ID_to_Rows_Dict[val].append(ind)
            else:
                Subj_ID_to_Rows_Dict[val] = [ind]
                
        # Subj_ID_to_Rows_Dict = {} # map ID to rows
        # for k in Subj_IDs_Unique:
        #     Subj_ID_to_Rows_Dict[k] = np.where(Subj_IDs == k)[0] 

        # Split
        Train_Inds, Val_Inds, Test_Inds = Support_Functions.Data_Split_Rand( [k for k in range(len(Subj_IDs_Unique))], TR, VA, TE)

        # Speedup 08/14/24
        Train_Inds_ECG  = [Row for PID in Train_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        Val_Inds_ECG  = [Row for PID in Val_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        # Test_Inds_ECG = [Row for PID in Test_Inds for Row in Subj_ID_to_Rows_Dict[Subj_IDs_Unique[PID]] ]
        
        Data['x_valid'] = Data['x_train'][Val_Inds_ECG]
        Data['x_train'] = Data['x_train'][Train_Inds_ECG]
    
        Data['y_valid'] = Data['y_train'][Val_Inds_ECG]
        Data['y_train'] = Data['y_train'][Train_Inds_ECG]
          
    else:
        print('No Data Split')
    
    print('Split Train into Train/Valid. Total time elapsed: ' + str(time.time()-start_time) )                            
        
    # %% 9. set up trained model folders if they  don't exist

    # trained models live in /trained_models/[train_data_folder]/Resnet18_Bob/asdf.pt
    # evals live in /trained_models/[train_data_folder]/Resnet18_Bob/Eval/[Eval_data_folder]/output.csv    
    
    # Set up Trained_Models folder
    temp_path = os.path.join(os.getcwd(),'Trained_Models')
    if ( (os.path.exists(temp_path) == False)):
        os.mkdir(temp_path)
    
    # Set up Trained_Models/[Train_Folder]
    temp_path = os.path.join(temp_path, args['Train_Folder'])
    if ( (os.path.exists(temp_path) == False)):
        os.mkdir(temp_path)
        
    # Set up and remember Trained_Models/[Train Folder]/[Model_Name]
    Model_Folder_Path = os.path.join(temp_path, args['Model_Name'])
    if (os.path.exists(Model_Folder_Path) == False):
        os.mkdir(Model_Folder_Path)
    
    args['Model_Folder_Path'] = Model_Folder_Path # pass model folder path to model to save out checkpoints / eval
    
    
    # %% 10. Select model, (maybe) train, and run
    # # Speedup debugging?
    if (debug):
        print('!!!debug!!! Reducing data size!')
        breakpoint()
        tr_inds = np.random.randint(0, Data['x_train'].shape[0], (sub_len))
        va_inds = np.random.randint(0, Data['x_valid'].shape[0], (sub_len))
        te_inds = np.random.randint(0, Data['x_test'].shape[0], (sub_len))
        
        Data['x_train'] = Data['x_train'][tr_inds,:]
        Data['x_valid'] = Data['x_valid'][va_inds,:]
        Data['x_test'] = Data['x_test'][te_inds,:]
        
        Data['y_train'] = Data['y_train'][tr_inds]
        Data['y_valid'] = Data['y_valid'][va_inds]
        Data['y_test'] = Data['y_test'][te_inds]
        
        
    # %% 10. Select model, (maybe) train, and run
         
    print('Got to init. Total time elapsed: ' + str(time.time()-start_time) )
    if (Model_Type == 'InceptionClass'):
        asdf = InceptionTimeClassifier.InceptionTimeClassifier(args, Data)
    if (Model_Type == 'RibeiroClass'):
        asdf = Ribeiro_Classifier.Ribeiro_Classifier(args, Data)
        
    if (Model_Type == 'Resnet18'):
        asdf = CV_Resnet18_Classifier_Flip.CV_Resnet18_Classifier_Flip(args, Data)
        
    if (Model_Type == 'FFClass'):
        asdf = FFClass.FFClass(args, Data)
        
    if (Model_Type == 'LSTMClass'):
        asdf = LSTMClass.LSTMClass(args, Data)
        
    if (Model_Type == 'TimesNetClass'):        
        asdf = TimesNetClass.TimesNetClass(args, Data)
        
    print('Got to Train. Total time elapsed: ' + str(time.time()-start_time) )
    if( ('Load' in args.keys())):
        asdf.Load(args['Load'])
    asdf.Train()
        

    # %% 11. Generate and save out results    
    if ('Test_Folder' in args.keys()):
        print('got to eval. Total Time elapsed: ' + str(time.time()-start_time))
        
        # get test, validation outputs
        if ('Eval_Dataloader' in args.keys()): # This lets you evaluate the model on its validation set instead of test set
            test_outputs, test_Loss, test_correct_outputs = asdf.Test(Which_Dataloader = args['Eval_Dataloader'])
        else:
            test_outputs, test_Loss, test_correct_outputs = asdf.Test()
        val_outputs, val_Loss, val_correct_outputs    = asdf.Test(Which_Dataloader = 'Validation')
    
        # adjust output formats
        val_outputs  = np.squeeze(val_outputs)
        test_outputs = np.squeeze(test_outputs)
        test_correct_outputs = np.squeeze(np.array(test_correct_outputs))
        val_correct_outputs = np.squeeze(np.array(val_correct_outputs))

        # %%
        # 12. Set up folders.
        # evals live in /trained_models/[train_data_folder]/Resnet18_Bob/Eval/[eval_data_folder]/output.csv
        
        # Set up .../Eval
        temp_path = os.path.join(Model_Folder_Path, 'EVAL')
        if (os.path.isdir(temp_path) == False):
            os.mkdir(temp_path)
            
        # Set up .../Eval/eval_data_folder
        temp_path = os.path.join(temp_path, args['Test_Folder'] + ' Test_Folder')  
        if (os.path.isdir(temp_path) == False):
            os.mkdir(temp_path)
            
        # Save out Evaluation Args
        path = os.path.join(temp_path, 'Eval_Args.txt')
        with open(path, 'w') as file:
             file.write(json.dumps(args)) # use `json.loads` to do the reverse
             
        # Plot argmax == correct
        # fig1, ax = plt.subplots()
        # if len(test_outputs.shape) > 1: # if this is a classifier, return argmax
        #     test_argmax = np.array(test_outputs).argmax(axis=1) # argmax
            
        #     # let's do accuracy per class?
        #     ACC = [ sum([1 for k in range(len(test_argmax)) if (test_argmax[k]==m and test_argmax[k]==test_correct_outputs[k])]) / sum([1 for k in range(len(test_correct_outputs)) if test_correct_outputs[k] == m]) for m in np.unique(test_correct_outputs)]
        #     ax.bar(np.unique(test_correct_outputs),ACC)
        #     ax.set(xlabel='correct output', ylabel = 'accuracy')
        #     print("accuracy per class: ")
        #     print ( [k for k in zip(np.unique(test_correct_outputs),np.array(ACC))] )
        #     print("\n")
        # else:
        #     ax.scatter(test_correct_outputs, test_outputs)
        #     ax.set(xlabel='correct output', ylabel = 'predicted output')
        
        # ax.set_title(args['Model_Name'])
        # plot_file_path = os.path.join(temp_path, 'Classifier Output.pdf')
        # fig1.savefig(plot_file_path)
        
        # %% 13. Run Cox models
        
        # since we care about survival modeling, we're going to fit a cox  model on the VALIDATION set
        # then evaluate it on the TEST set
        # 1. convert risk prediction (0-1)  to survival curves per subject
        # 2. measure concordance, brier, AUPRC, AUROC, over 100 time points

        # build CoxPH curves on validation data
        zcxv = CoxPHSurvivalAnalysis() 
        a = Data['y_valid'][:,[int(args['y_col_train_event']) ]].astype(bool)   # CoxPH is built on if/when the event actually happened
        b = Data['y_valid'][:,[int(args['y_col_train_time']) ]]                 # shouldn't this also be limited?
        tmp = np.array([ (a[k],b[k][0]) for k in range(a.shape[0]) ], dtype = [('event',bool),('time',float)] )
        val_outputs = np.array([softmax(k)[1] for k in val_outputs])
        zcxv.fit(np.expand_dims(val_outputs,-1), tmp   )
        
        # prep eval data - evaluate on if/when things actually happen
        if (args['Eval_Dataloader'] == 'Train'):
            disc_y_e = Data['y_train'][:,[int(args['y_col_train_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_train'][:,[int(args['y_col_train_time']) ]] # prep time
            
        elif (args['Eval_Dataloader'] == 'Validation'):
            disc_y_e = Data['y_valid'][:,[int(args['y_col_train_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_valid'][:,[int(args['y_col_train_time']) ]] # prep time
            
        else:
            disc_y_e = Data['y_test'][:,[int(args['y_col_test_event']) ]].astype(bool) # prep event
            disc_y_t = Data['y_test'][:,[int(args['y_col_test_time']) ]] # prep time
    
        test_outputs = np.array([softmax(k)[1] for k in test_outputs])
        

        # %% 14. Prep everything to match PyCox analysis
        # sample survival functions at a set of times (to compare to direct survival moels)
        upper_time_lim = max( b )[0] # the fit is limited to validation end times, so do 100 bins of tha
        sample_time_points = np.linspace(0, upper_time_lim, 100).squeeze()
        
        surv_funcs = zcxv.predict_survival_function(np.expand_dims(test_outputs,-1))
        surv = np.squeeze(  np.array([k(sample_time_points) for k in surv_funcs]))
        
        disc_y_e_bool = disc_y_e.squeeze()
        disc_y_e = disc_y_e.astype(int).squeeze()
        disc_y_t = np.array([np.argmax(sample_time_points>=k) if k<=upper_time_lim else len(sample_time_points)-1 for k in disc_y_t]) # bin times. none should appear in bin 0
        
        surv_df = pd.DataFrame(np.transpose(surv)) # already discretized, so concordance measures should go well
        
        # %% Save out sample_time_points, testy_t, testy_e, s(t)
        outputs_hdf5_path = os.path.join(temp_path, 'Stored_Model_Output.hdf5')
        Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(outputs_hdf5_path, disc_y_t, 'disc_y_t') # when it really happened
        Save_to_hdf5(outputs_hdf5_path, disc_y_e, 'disc_y_e') # what really happened
        Save_to_hdf5(outputs_hdf5_path, Data['y_test'], 'y_test')
        Save_to_hdf5(outputs_hdf5_path, surv, 'surv')
        Save_to_hdf5(outputs_hdf5_path, Test_Col_Names + ['PID', 'TTE*', 'E*'], 'Test_Col_Names')

        # %% 15. Save out KM. Add bootstrapping (20x)
        print('Got to KM bootstraps. Total time elapsed: ' + str(time.time()-start_time) )
        
        # Bootstrap
        km_outs = []
        mdl_outs = []
        for k in range(100):
            sample_inds = np.random.randint(0,surv.shape[0],(surv.shape[0]))
            km_out = kaplan_meier(events=disc_y_e[sample_inds], durations=disc_y_t[sample_inds]).to_numpy().tolist()
            mdl_out = np.mean(surv[sample_inds],axis=0)
            
            # sometimes don't get a last index on the KM - extend last value out
            while (len(km_out) < surv.shape[1]):
                km_out.append(km_out[-1])
                
            km_outs.append(km_out)
            mdl_outs.append(mdl_out)
            
        mdl_outs = np.vstack(mdl_outs)
        km_outs = np.vstack(km_outs)
        
        mdl_int_low = [] # 
        mdl_int_high = []
        mdl_median = []
        km_int_low = []
        km_int_high = []
        km_median = []
        for k in range(km_outs.shape[1]):
            km_median.append(np.median(km_outs[:,k]))
            km_int_high.append(np.percentile(km_outs[:,k],97.5))
            km_int_low.append(np.percentile(km_outs[:,k],2.5))
            mdl_median.append(np.median(mdl_outs[:,k]))
            mdl_int_high.append(np.percentile(mdl_outs[:,k],97.5))
            mdl_int_low.append(np.percentile(mdl_outs[:,k],2.5))
            
        fig1, ax = plt.subplots()
        plt.plot(sample_time_points, km_median)
        ax.fill_between(sample_time_points, km_int_low, km_int_high, color='b', alpha=.1)
        plt.plot(sample_time_points,mdl_median, color='r')
        ax.fill_between(sample_time_points, mdl_int_low, mdl_int_high, color='r', alpha=.1)
        plt.legend(('KM','KM 2.5-97.5%','Model','Model 2.5-97.5%'))
        plt.xlabel('Years')
        plt.ylabel('Survival')
        plot_file_path = os.path.join(temp_path, 'KM vs Model 100xBS Survival Curve.pdf')
        fig1.savefig(plot_file_path)
        
        
        outputs_hdf5_path = os.path.join(temp_path, 'Surv_Outputs.hdf5')
        Save_to_hdf5(outputs_hdf5_path, sample_time_points, 'sample_time_points')
        Save_to_hdf5(outputs_hdf5_path, mdl_median, 'SF_mdl_median')
        Save_to_hdf5(outputs_hdf5_path, mdl_int_low, 'SF_int_low')
        Save_to_hdf5(outputs_hdf5_path, mdl_int_high, 'SF_mdl_int_high')
        Save_to_hdf5(outputs_hdf5_path, km_median, 'SF_km_median')
        Save_to_hdf5(outputs_hdf5_path, km_int_low, 'SF_km_int_low')
        Save_to_hdf5(outputs_hdf5_path, km_int_high, 'SF_km_int_high')


        print('Finished KM bootstraps. Starting all-ECG briercordance Total time elapsed: ' + str(time.time()-start_time) )
        # %% 16. Concordance and Brier Score across all ECGs

        # discretize test_outputs time
        concordance_store_all_ecg, ipcw_brier_store_all_ecg, chance_at_censored_point_all_ecg  = get_surv_briercordance(disc_y_t, disc_y_e_bool, surv_df, [1,2,5,10,999], sample_time_points)
        
        Save_to_hdf5(outputs_hdf5_path, concordance_store_all_ecg, 'concordance_store_all_ecg')
        Save_to_hdf5(outputs_hdf5_path, ipcw_brier_store_all_ecg, 'ipcw_brier_store_all_ecg')
        Save_to_hdf5(outputs_hdf5_path, chance_at_censored_point_all_ecg, 'chance_at_censored_point_all_ecg')
        
        fig1, ax = plt.subplots()
        plt.plot([0,upper_time_lim],[.5,.5], '--') # concordance if guessing
        plt.plot(sample_time_points, ipcw_brier_store_all_ecg)
        plt.plot(sample_time_points, concordance_store_all_ecg)
        plt.ylim([0,1])
        plt.legend(['Concordance Chance','Brier = MSE','Concordance'])
        plt.xlabel('follow-up time (years)')
        plt.ylabel('Measure for time bin')
        plot_file_path = os.path.join(temp_path, 'briercordance, briercordance.pdf')
        fig1.savefig(plot_file_path)

        
        # %% 17. Concordance and Brier, bootstrapping 1 ECG per patient
        # (later, they will be combined for all the random seeds into a single mean / stdev / 95% CI)
        print('Got to bootstrap concordances. Total time elapsed: ' + str(time.time()-start_time) )
        # 1. find relevant rows per subjetc ID
        if (args['Eval_Dataloader'] == 'Validation'):
            Subj_IDs = Data['y_valid'][:,-3]    # PID lives in -3
        elif (args['Eval_Dataloader'] == 'Train'):
            Subj_IDs = Data['y_train'][:,-3]    
        else:
            Subj_IDs = Data['y_test'][:,-3]  
            
        Subj_IDs_Unique = np.unique(Subj_IDs)
        Subj_ID_to_Rows_Dict = {} # map ID to rows
        for ind,val in enumerate(Subj_IDs):
            if val in Subj_ID_to_Rows_Dict.keys():
                Subj_ID_to_Rows_Dict[val].append(ind)
            else:
                Subj_ID_to_Rows_Dict[val] = [ind]
            
        bootstrap_briers = [] # list of lists
        bootstrap_concordances = [] # list of lists
        
        bootstraps = 20 
        Inds = [Subj_ID_to_Rows_Dict[k][0] for k in Subj_IDs_Unique]
        
        for b in range (bootstraps):
            # 3. Sample one revelant Surv row per each subject.
            for i,s in enumerate(Subj_IDs_Unique):
                tmp = Subj_ID_to_Rows_Dict[s]
                if (len(tmp) != 1):
                    Inds[i] = tmp[np.random.randint(0,len(tmp))]
            
            concordance_score, ipcw_brier_score, chance_at_censored_point  = get_surv_briercordance(disc_y_t[Inds], disc_y_e[Inds], surv_df.iloc[:,Inds], [1,2,5,10,999], sample_time_points)
            
            bootstrap_briers.append(ipcw_brier_score)
            bootstrap_concordances.append(concordance_score)
                
        Save_to_hdf5(outputs_hdf5_path, bootstrap_briers, 'bootstrap_briers')
        Save_to_hdf5(outputs_hdf5_path, bootstrap_concordances, 'bootstrap_concordances')
        print('Finished bootstrap concordance. Heading to AUC Total time elapsed: ' + str(time.time()-start_time) )

        # %%  18. Now add AUROC and AUPRC of the survival output
        S_AUROC_store, S_AUPRC_store, Chance_At_Age = get_AUROC_AUPRC(disc_y_t, disc_y_e, surv, [1,2,5,10], sample_time_points)
        
        # how did the classifier perform at its classification task?
        Classif_AUROC = roc_auc_score(test_correct_outputs,test_outputs)
        Classif_AUPRC = average_precision_score(test_correct_outputs,test_outputs)
        
        print('Classif_AUROC ' + str(Classif_AUROC))
        print('Classif_AUPRC ' + str(Classif_AUPRC))
        
        fig1, ax = plt.subplots()
        plt.plot([0,upper_time_lim],[.5,.5], '--')
        plt.plot(sample_time_points,Chance_At_Age,'--')
        plt.plot(sample_time_points, S_AUROC_store)
        plt.plot(sample_time_points, S_AUPRC_store)
        plt.ylim([0,1])
        plt.legend(['AUROC Chance' , 'Chance at time','AUROC','AUPRC'])
        plt.xlabel('follow-up time (years)')
        plt.ylabel('Measure for time bin')
        plot_file_path = os.path.join(temp_path, 'AUROC, AUPRC.pdf')
        fig1.savefig(plot_file_path)
        
        Save_to_hdf5(outputs_hdf5_path, S_AUROC_store, 'AUROC')
        Save_to_hdf5(outputs_hdf5_path, S_AUPRC_store, 'AUPRC')
        
        # %% 19. Now save out a histogram as csv
        
        fig1, ax = plt.subplots(2)
        quant, bin_loc = np.histogram(sample_time_points[disc_y_t],bins=surv.shape[1])
        ax[1].bar(bin_loc[1:],quant,width= (max(sample_time_points)-min(sample_time_points))/len(sample_time_points))
        ax[1].set(xlabel = 'Time to event or censor (years)' , ylabel = 'Sample Count' )
        plot_file_path = os.path.join(temp_path, 'Time Dist Histogram.pdf')
        fig1.savefig(plot_file_path)

        hist_path = os.path.join(temp_path, 'Histogram.csv')
        temp = np.transpose(np.vstack( (bin_loc[1:], quant, quant[0] + sum(quant) - np.cumsum(quant))))
        headers = "bin end time, quantity in bin, quantity at risk "
        np.savetxt(hist_path, temp, header=headers, delimiter = ',')
        
        print('Finished evaluation. Total time elapsed: ' + str(time.time()-start_time) )
        
        
        # %% multimodal work
        # # %% 20. Save out some more things 
        # multim_path = os.path.join(temp_path, 'Multimodal_Out.csv')
        # temp = np.vstack( (Data_Y_Backup['y_test'][:,0],test_outputs,test_correct_outputs,Data_Y_Backup['y_test'][:,3], Data_Y_Backup['y_test'][:,4] )  )
        # temp = np.transpose(temp)
        # headers = 'PID, test_outputs, test_correct_output, TTE, E'
        # np.savetxt(multim_path, temp, header=headers, delimiter = ',')
        
        # auroc_2, auprc_2, Chance_At_Age = get_AUROC_AUPRC(disc_y_t, disc_y_e, surv, [30/365], sample_time_points)
        
        
        # # PID = Data_Y_Backup['y_test'][:,0]
        # # test_outputs # softmax
        # # test_correct_outputs # label passed
        # # TTE = Data_Y_Backup['y_test'][:,3]
        # # TTE = Data_Y_Backup['y_test'][:,4]
        
        # # %%21. Save out some more details
        
        # # 1. What is distribution of ECG count per PID in last T time?
        # x = 0.05*np.arange(21)
        # save_out = [x]
        
        # for time_lim in [1/52,1/12,1,999]:
        #     subset = Data_Y_Backup['y_train'][np.where(Data_Y_Backup['y_train'][:,3] <= time_lim)[0],:]
        #     unique_PID = np.unique(subset[:,0])
        #     counts = [np.where(subset[:,0] == k)[0].shape[0] for k in unique_PID]
        #     event = [Data_Y_Backup['y_train'][np.where(subset == k)[0][0],4] for k in unique_PID  ]
        #     event = np.array(event).astype(int)
        #     counts = np.array(counts).astype(int)
            
        #     y1 = np.quantile(counts[np.where(event==1)],x)
        #     y2 = np.quantile(counts[np.where(event==0)],x)
        #     save_out.append(y1)
        #     save_out.append(y2)
            
        #     plt.figure()
        #     plt.scatter(x*100,y2)
        #     plt.scatter(x*100,y1)
        #     plt.legend(['event=0 med '+str(np.median(y2))+' u '+'{0:.2f}'.format(np.mean(y2)),'event=1 med '+str(np.median(y1)) + ' u '+'{0:.2f}'.format(np.mean(y1))])
        #     plt.xlabel('percentile')
        #     plt.ylabel('count')
        #     plt.title('count per PID | TTE < '+str(time_lim))
            
        # characteristics_path = os.path.join(temp_path, 'Train_Characteristics.csv')
        # headers = 'percentile, 1-wk event 1, 1-wk event 0, 1-mo event 1, 1-mo event 0,1-yr event 1, 1-yr event 0, all-t event 1, all-t event 0'
        # temp = np.array(save_out).transpose()
        # np.savetxt(characteristics_path, temp, header=headers, delimiter = ',')
        
        
        
    # %% load?
    # asdf.Load()
    
    #%% Test?
if __name__ == '__main__':
   main()