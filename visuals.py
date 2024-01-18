# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 00:33:40 2024

@author: Lalith_B
"""
import matplotlib.pyplot as plt
import numpy as np

img  = np.random.randint(0, 255, size=(128, 128, 1))
mask = np.random.randint(0, 255, size=(128, 128, 1))



def create_image_and_mask_plot(img= img,
                               mask = mask):
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
    
    
    
    
