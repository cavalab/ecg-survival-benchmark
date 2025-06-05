import torch
import torch.nn as nn
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, Input

# %%
class ReservoirMLP(nn.Module):
    # input: ECG, B x ...
    # output: run that through the reservoir, then an NLP
    
    def __init__(self):
        super(ReservoirMLP, self).__init__()
        
        # set up reservoir computing
        data = Input(input_dim=12)
        reservoir = Reservoir (1024, lr = 0.3, sr = 1.1) # compress 4096x12 into 1024
        self.ResComp = data >> reservoir 

            
    def forward(self, x):
        outs = self.ResComp.run(x.cpu().numpy())
        outs = np.array(outs)
        outs = outs[:,-1,:].astype(np.float32)
        return torch.tensor(outs, device='cuda')
        # return torch.full([x.shape[0],1], self.const_value, device = x.device)
            
def get_ReservoirMLP():
    return ReservoirMLP()
    
             
# %% image processing
def Adjust_Image(single_image):
    # single_image = torch.transpose(single_image, 1,2)
    return single_image[0] # Just chan x leng, so 12 x 4k

def Adjust_Many_Images(image_batch):
    # This function is called after the image_batch is sent to GPU
    # image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
    return image_batch

def get_ReservoirMLP_process_single_image():
    return Adjust_Image

def get_ReservoirMLP_process_multi_image():
    return Adjust_Many_Images








    