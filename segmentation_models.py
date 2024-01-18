# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:34:56 2024

@author: Lalith_B
"""

import segmentation_models_pytorch as smp
from preprocessing import images
import torch
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt


model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1).cuda()
                      
summary(model, (1,512,512))
imgs = np.array(images, dtype=np.float32)
img = torch.from_numpy(imgs).cuda()
img1 = torch.reshape(img, (46, 1, 512, 512)).cuda()

predict = model(img1[20:27]).cuda()



def create_image_and_mask_plot(img,mask):
    #------2 plots in a single frame
    fig, axes = plt.subplots(1, 2, figsize=(8, 8), dpi=100)
    
    #------The Image in the Left
    axes[0].imshow(img, cmap='gray')  # You can customize the colormap as needed
    axes[0].set_title('Image')
    
    #-------Image's corresponding mask
    axes[1].imshow(mask, cmap='gray')  # You can customize the colormap as needed
    axes[1].set_title('Mask prediction')
    
    #-------Red Grids for gray images (medical images, generic case)
    for ax in axes:
        ax.grid(True,color='r')
    
    plt.tight_layout()
    plt.show()
    
        
for i in range(5):
    imge = torch.reshape(img[i+20], (512,512)).cpu().detach().numpy()
    mask= torch.reshape(predict[i], (512,512)).cpu().detach().numpy()
    create_image_and_mask_plot(imge, mask)