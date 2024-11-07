import torch
import torch.nn as nn

# %%
class ConstantNet(nn.Module):
    # input: N x whatever
    # output: N x 1, all values constant
    
    def __init__(self, const_value):
        super(ConstantNet, self).__init__()
        self.const_value = const_value
            
    def forward(self, x):
        return torch.full([x.shape[0],1], self.const_value, device = x.device)
            
def get_ConstantNet_model(const_value):
    return ConstantNet(const_value)
    
             
# %% image processing
def Adjust_Image(single_image):
    # single_image = torch.transpose(single_image, 1,2)
    return single_image[0] # Just chan x leng, so 12 x 4k

def Adjust_Many_Images(image_batch):
    # This function is called after the image_batch is sent to GPU
    # image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
    return image_batch

def get_ConstantNet_process_single_image():
    return Adjust_Image

def get_ConstantNet_process_multi_image():
    return Adjust_Many_Images








    