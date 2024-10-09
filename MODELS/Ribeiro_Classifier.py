

from MODELS.Ribeiro_Support import ResNet1d
import torch

def get_ribeiro_model(args, input_channels):
    print('Ribeiro Classifier currently expects signals to have length 4096')
    
    # 1. Process args to figure out how to set up the ribeiro classifier
    if ('seq_length' in args.keys()):
        seq_length = int(args['seq_length'])
    else:
        seq_length = 4096
        
    if ('sample_freq' in args.keys()):
        sample_freq = int(args['sample_freq'])
    else:
        sample_freq = 400
        
    if ('scale_multiplier' in args.keys()): # help='multiplicative factor used to rescale inputs.')
        scale_multiplier = int(args['scale_multiplier'])
    else:
        scale_multiplier = 10
        
    # ... here we're going to hard-code net filter size, cause we can't get that from a set of strings
    net_filter_size = [64, 128, 196, 256, 320] #'filter size in resnet layers (default: [64, 128, 196, 256, 320]).'
    net_seq_lengh = [4096, 1024, 256, 64, 16] #'number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).'
        
    if ('dropout_rate' in args.keys()): #help='reducing factor for the lr in a plateu (default: 0.1)')
        dropout_rate = float(args['dropout_rate'])
    else:
        dropout_rate = 0.8
    
    if ('kernel_size' in args.keys()):
        kernel_size = int(args['kernel_size'])
    else:
        kernel_size = 17
        
    # --
    
    model = ResNet1d(input_dim=(input_channels, seq_length),
                     blocks_dim=list(zip(net_filter_size, net_seq_lengh)),
                     n_classes=0,
                     kernel_size=kernel_size,
                     dropout_rate=dropout_rate)
    print('Ribeiro_Classifier: ','Returning Features')
    
    return model
    
             
# %% image processing
def Adjust_Image(single_image):
    single_image = torch.transpose(single_image, 1,2)
    return single_image[0] # Just chan x leng, so 12 x 4k

def Adjust_Many_Images(image_batch):
    # This function is called after the image_batch is sent to GPU
    image_batch = torch.transpose(image_batch[:,0,:,:],1,2) # This model wants data N-Chan-Len
    return image_batch

def get_ribeiro_process_single_image():
    return Adjust_Image

def get_ribeiro_process_multi_image():
    return Adjust_Many_Images








    