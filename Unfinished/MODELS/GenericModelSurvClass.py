"""
This is a generic SurvClass model:
    
    Data input is [ecg], [E*], where E* is built from (TTE,E)
    From this classifier output, we then fit Cox regressions on validation data
    And those ultimately provide the survival curves on the test set
    
New policy 09/10/24:
    Generic models have their own dataset / shuffle / collate functions
    And anything else they need to run
    
"""

import torch
import numpy as np
import time
import os
import copy

from MODELS.GenericModel import GenericModel

from MODELS.Support_Functions import Get_Loss
from MODELS.Support_Functions import Save_NN
from MODELS.Support_Functions import Save_Train_Args


# models
from MODELS.Ribeiro_Classifier import get_ribeiro_model
from MODELS.Ribeiro_Classifier import get_ribeiro_process_multi_image

from MODELS.InceptionTimeClassifier import get_InceptionTime_model
from MODELS.InceptionTimeClassifier import get_InceptionTime_process_multi_image



# %% Datasets, Samplers, and Collate functions
class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, more_data, targets):
        self.data = data
        self.more_data = torch.tensor(more_data)
        self.targets = targets
        # self.ones = torch.ones(1,4096,1)
        
    def __getitem__(self, index):
        x = self.data[index] # if you ever modify this value, torch.clone it first (else permanent)        
        y = self.targets[index]        
        z = self.more_data[index]
        # out_x = torch.concat((x, self.ones*z),dim=2) # 200us (240 without preallocating ones)
        return x, y, z
    
    def __len__(self):
        return self.data.shape[0]
    
def Custom_Collate(batch):
    # here we return ecg, cov, and correct_label
    ecg_out = torch.stack([k[0] for k in batch]) # 32 x 1 x 4096 x 12
    event_label = torch.tensor(np.array([k[1] for k in batch] ))
    cov = torch.stack([k[2] for k in batch])
    # if (len(batch[0][2]) > 0):
    #     cov = torch.stack([k[2] for k in batch],axis=0 ) # 32 x 2.  # we want to expand these to 32 x 4096 x 2
    #     cov = cov.unsqueeze(1).unsqueeze(1)
    #     app = cov.repeat([1,1,4096,1])
    #     ecg_out = torch.concat((ecg_out, app),dim=3)
    return ecg_out, event_label, cov


# %%  ---------------- Start the model
class GenericModelSurvClass(GenericModel):

    def __init__(self, args, Data):
        
        # 0. Process input arguments 
        self.Process_Args(args)
        self.Data = Data
        
        # 1. Adjust data for this model
        self.restructure_data()
        self.prep_normalization_parameters()
        self.normalize_data()
        
        # 2. Prepare dataloaders
        self.prep_dataloaders()
        
        # 3. Prep Loss Params
        self.prep_classif_loss()

            
        # 4. Grab ECG processing network 
        self.gen_ecg_model()
            
        # 5. wrap our network with the fusion modules before building optimizer/scheduler
        if ('direct' in args.keys()):
            if (args['direct'] == 'True'):
                self.prep_fusion(out_classes = 2, direct = True)
            else:
                self.prep_fusion(out_classes = 2, direct = False)
        else:
            self.args['direct'] = 'False'
            self.prep_fusion(out_classes = 2, direct = False)
        
            
        # 6. Optimizer and scheduler
        self.prep_optimizer_and_scheduler()
        
    
# %% Init functions
    def gen_ecg_model(self):
        # figures out how to summon model, image adjustment functions
        
        breakpoint()
        # 1 figure out output channel size
        if 'x_train' in self.Data.keys():
            n_in_channels = self.Data['x_train'].shape[-1]
        else:
            n_in_channels = 12
        
        if (self.args['Model_Type'] == 'RibeiroClass'):
            self.model = get_ribeiro_model(self.args, n_in_channels) # get the ECG interpreting model
            self.Adjust_Many_Images = get_ribeiro_process_multi_image() # pointer to function
            
        if (self.args['Model_Type'] == 'InceptionTimeClass'):
            self.model = get_InceptionTime_model(self.args, n_in_channels) # get the ECG interpreting model
            self.Adjust_Many_Images = get_InceptionTime_process_multi_image() # pointer to function

        
    def prep_dataloaders(self):
        a = time.time()
        if 'x_train' in self.Data.keys():
            self.train_dataset = Custom_Dataset( self.Data['x_train'] , self.Data['z_train'], self.Data['y_train'][:,-1]) # modified event, e*, lives in -1
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.GPU_minibatch_limit, collate_fn=Custom_Collate, shuffle = True) # weighted sampler is mutually exclussive with shuffle = True
            
        if 'x_valid' in self.Data.keys():
            self.val_dataset = Custom_Dataset(self.Data['x_valid']  , self.Data['z_valid'], self.Data['y_valid'][:,-1]) # modified event, e*, lives in -1
            self.val_dataloader = torch.utils.data.DataLoader (self.val_dataset,  batch_size = self.GPU_minibatch_limit, collate_fn=Custom_Collate, shuffle = False) #DO NOT SHUFFLE

        if 'x_test' in self.Data.keys():
            self.test_dataset  = Custom_Dataset(self.Data['x_test']  , self.Data['z_test'], self.Data['y_test'][:,-1]) # modified event, e*, lives in -1
            self.test_dataloader = torch.utils.data.DataLoader (self.test_dataset,  batch_size = self.GPU_minibatch_limit, collate_fn=Custom_Collate, shuffle = False) #DO NOT SHUFFLE
        print('Generic_Model_SurvClass: prep_dataloaders T = ', '{:.2f}'.format(time.time()-a))


# %% Training
    def train(self):
        Best_Model = copy.deepcopy(self.model) # need to start with 'some' best model
        
        train_loss = -1 # in case no training occurs
        accumulation_iterations = int(self.batch_size / self.GPU_minibatch_limit)
        
        # Try to load a checkpointed model?
        if self.epoch_end > self.epoch_start:
            print('GenericModel.Train(): Training Requested. Loading best then last checkpoints.')
            last_checkpoint_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
            best_checkpoint_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
            if (os.path.isfile(last_checkpoint_path)):
                if (os.path.isfile(best_checkpoint_path)):
                    self.Load('Best')
                    Best_Model = copy.deepcopy(self.model)
                    print('GenericModel.Train(): Best Checkpoint loaded and Best model copied.')
                    self.Load('Last')
                    print('GenericModel.Train(): Checkpointed model loaded. Will resume training.')
                    
                    val_perfs = np.array([k[2] for k in self.Perf])
                    if (self.early_stop > 0):
                        if (len(val_perfs) - (np.argmin(val_perfs) + 1 ) ) >= self.early_stop:
                            # ^ add one: len_val_perfs is num trained epochs (starts at 1), but argmin starts at 0.
                            print('GenericModel.Train(): Model at early stop. Setting epoch_start to epoch_end to cancel training')
                            self.epoch_start = self.epoch_end
                            
                    if (self.epoch_start == self.epoch_end):
                        print('GenericModel.Train(): Loaded checkpointed model already trained')
                    if (self.epoch_start > self.epoch_end):
                        print('GenericModel.Train(): Requested train epochs > current epochs trained. evaluating.')
                        self.epoch_start = self.epoch_end
                else:
                    print('GenericModel.Train(): FAILED to load best model! Eval may be compromised')
            else:
                print('GenericModel.Train(): Last checkpoint unavailable.')
        
        # If more training requested, train
        for epoch in range(self.epoch_start, self.epoch_end):
            self.model.train()
            epoch_start_time = time.time()
            
            train_loss = 0
            for i, (imgs , labels, cov) in enumerate(self.train_dataloader):

                imgs = imgs.to(self.device)
                imgs = imgs.to(torch.float32) # convert to float32 AFTER putting on GPU
                # imgs = Normalize(imgs, self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
                imgs = self.Adjust_Many_Images(imgs) # adjust a batch of ECGs to fit the ECG model
                
                labels = labels.to(self.device)
                labels = labels.to(torch.float32) # again, convert type AFTER putting on GPU
                
                cov = cov.to(self.device)
                cov = cov.to(torch.float32)

                model_out = self.model(imgs, cov)
                
                
                loss = Get_Loss(model_out, labels, self.Loss_Params) 
                train_loss += loss.item()
                loss.backward()
                
                # minibatch implem - from https://kozodoi.me/blog/20210219/gradient-accumulation
                if  ( ((i + 1) % accumulation_iterations ==0) or (i+1) == len(self.train_dataloader)):
                    self.optimizer.step() 
                    self.model.zero_grad()  

            epoch_end_time = time.time()

            # ----
            # Run Validation and Checkpoint
            if ( (epoch+1) % self.validate_every ==0):
                
                outputs, val_loss, correct_outputs = self.Run_NN(self.val_dataloader)
                
                # update scheduler # no effect unless args['Scheduler'] == 'True'
                if (hasattr(self,'scheduler')):
                    self.scheduler.step(val_loss)
                    tmp_LR = self.optimizer.state_dict()['param_groups'][0]['lr']
                else:
                    tmp_LR = 0
                    
                # If this is the new best model, save it as the best model
                if val_loss < self.Val_Best_Loss: 
                    nn_file_path = os.path.join(self.model_folder_path, 'Best_Checkpoint.pt')
                    if (self.Save_Out_Best):
                        Save_NN(epoch, self.model, nn_file_path, optimizer=None, scheduler=None, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev)
                        Best_Model = copy.deepcopy(self.model) # store a local copy of the model
                    Save_Train_Args(os.path.join(self.model_folder_path,'Train_Args.txt'), self.args)
                    self.Val_Best_Loss = val_loss
                    
                # And checkpoint model in any case
                nn_file_path = os.path.join(self.model_folder_path, 'Checkpoint.pt')
                if (self.Save_Out_Checkpoint):
                    if (hasattr(self,'scheduler')):
                        Save_NN(epoch, self.model, nn_file_path, optimizer = self.optimizer, scheduler=self.scheduler, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev)
                    else:
                        Save_NN(epoch, self.model, nn_file_path, optimizer = self.optimizer, scheduler=None, best_performance_measure = val_loss, NT = self.Normalize_Type, NM = self.Normalize_Mean, NS = self.Normalize_StDev)
                Save_Train_Args(os.path.join(self.model_folder_path,'Train_Args.txt'), self.args)

                # calculate new performance
                new_perf = [epoch, train_loss, val_loss, tmp_LR, epoch_end_time - epoch_start_time]
                print(new_perf)
                self.Perf.append(new_perf)
                
                # update performance                
                csv_file_path = os.path.join(self.model_folder_path, 'Training_Progress.csv')
                np.savetxt(csv_file_path, np.asarray(self.Perf), header = "Epoch,Train Loss, Validation Loss, LR, Runtime seconds",delimiter = ',')
                
                # save out performance curves
                Perf_Plot_Path = os.path.join(self.model_folder_path, 'Training_Plot.png')
                self.Save_Perf_Curves(self.Perf, Perf_Plot_Path)
                
                # consider stopping
                val_perfs = np.array([k[2] for k in self.Perf])
                if (self.early_stop > 0):
                    if (len(val_perfs) - (np.argmin(val_perfs) + 1 ) ) >= self.early_stop:
                        # ^ add one: len_val_perfs is num trained epochs (starts at 1), but argmin starts at 0.
                        break
                     
        # now that we're done training, load the best model back for evaluation
        self.model = copy.deepcopy(Best_Model)
        return train_loss


# %% Model Running
    def Run_NN (self, my_dataloader):
        # Runs the net in eval mode, predicted output, loss, correct output
        self.model.eval()
        
        tot_loss = 0
        
        outputs = []
        correct_outputs = [] 
        
        for i, (imgs , labels, cov) in enumerate(my_dataloader):
            imgs = imgs.to(self.device)
            imgs = imgs.to(torch.float32) # convert to float32 AFTER putting on GPU
            # imgs = Normalize(imgs, self.Normalize_Type, self.Normalize_Mean, self.Normalize_StDev)
            imgs = self.Adjust_Many_Images(imgs)
            
            labels = labels.to(self.device)
            labels = labels.type(torch.float32) # again, convert type AFTER putting on GPU
            
            cov = cov.to(self.device)
            cov = cov.to(torch.float32)

            with torch.no_grad():
                model_out = self.model(imgs, cov)
        
            loss = Get_Loss(model_out, labels, self.Loss_Params) 
        
            tot_loss += loss.item()
            outputs = outputs + model_out.to("cpu").detach().tolist()
            correct_outputs = correct_outputs + labels.to("cpu").detach().tolist()
        
        return outputs, tot_loss, correct_outputs
    
    # %% Test
    def Test(self, Which_Dataloader = 'Test'):
        if (Which_Dataloader == 'Train'):            
            self.train_dataloader.batch_sampler.shuffle = False
            outputs, test_loss, correct_outputs = self.Run_NN(self.train_dataloader) # NOT shuffled
            self.train_dataloader.batch_sampler.shuffle = True
        elif (Which_Dataloader == 'Validation'):
            outputs, test_loss, correct_outputs = self.Run_NN(self.val_dataloader) # NOT shuffled
        else:
            outputs, test_loss, correct_outputs = self.Run_NN(self.test_dataloader) # NOT shuffled
        return outputs, test_loss, correct_outputs


# %% Model Loading
    def Load(self, best_or_last):
        
        Import_Dict = self.Load_Checkpoint(best_or_last)   
        self.Load_Training_Params(Import_Dict)
        self.Load_Training_Progress(Import_Dict)
        self.Load_Normalization(Import_Dict) # we're frontloading normalization, so that doesn't matter

        self.model.load_state_dict(Import_Dict['model_state_dict'])
        
        if ('optimizer_state_dict' in Import_Dict.keys()):
            self.optimizer.load_state_dict(Import_Dict['optimizer_state_dict'])
        else:
            print('NO optimizer loaded')
        if ('scheduler_state_dict' in Import_Dict.keys()):
            self.scheduler.load_state_dict(Import_Dict['scheduler_state_dict'])
        else:
            print("NO scheduler loaded")

        self.Load_Random_State(Import_Dict)


    
