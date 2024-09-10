# InceptionTImeClassifier is a Generic_Model
# args['Loss Type'] defaults to CrossEntropyLoss
# adapted from
# https://github.com/TheMrGhostman/InceptionTime-Pytorch/commit/2a1b81d82b624c9033e73a2e7ee9bba2414217ef
# Heavily modified by PVL to line up with the rest of the survival modeling flow
# Where possible, the original code was kept

import torch
from MODELS.InceptionTime_Support import Inception, InceptionBlock, InceptionTime


def get_InceptionTime_model(args, input_channels):
    
    # not presently implemented
    # if ('K_M' in args.keys()):
    #     K_M = int(args['K_M'])
    # else:
    #     K_M = 1
    #     print('defaulting to kernel mult of 1. InceptionTime kernel sizes are 10,20,40')
        
    model = InceptionTime(in_channels=input_channels, out_channels=0)
    
    return model

    
# %% image processing
def Adjust_Image( single_image):
    single_image = torch.transpose(single_image, 1,2)
    return single_image[0] # Just chan x leng, so 12 x 4k

def Adjust_Many_Images(image_batch):
    # This function is called after the image_batch is sent to GPU
    image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
    return image_batch

def get_InceptionTime_process_single_image():
    return Adjust_Image

def get_InceptionTime_process_multi_image():
    return Adjust_Many_Images

