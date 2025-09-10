#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN Encoder module for image captioning.
This module implements the encoder part of the image captioning system.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder for extracting feature representations from images.
    Uses a pre-trained CNN backbone with the classification head removed.
    """
    
    def __init__(self, model_name='resnet18', embed_size=256,grid_size = 2 , overlap_ratio = 0.25,  pretrained=True, trainable=False , project = True ):
        """
        Initialize the encoder.
        
        Args:
            model_name (str): Name of the CNN backbone to use
                Supported models: 'resnet18', 'resnet50', 'mobilenet_v2', 'inception_v3'
            embed_size (int): Dimensionality of the output embeddings
            pretrained (bool): Whether to use pre-trained weights
            trainable (bool): Whether to fine-tune the CNN backbone
        """
        super(EncoderCNN, self).__init__()
        self.model_name = model_name.lower()
        self.embed_size =embed_size
        self.project = project
        self.grid_size = grid_size 
        self.overlap_ratio = overlap_ratio
        if self.model_name == 'inception_v3':
            model = models.inception_v3(weights = pretrained ) 
    
            model.AuxLogits = nn.Identity()
            #model.dropout = nn.Identity()
            #model.fc = nn.Identity()
            self.cnn = torch.nn.Sequential(
                *(list(model.children())[:-2]),  # Skip Dropout, and FC
                nn.Flatten()
            )
            
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained = pretrained )
            model.classifier = nn.Identity()
            self.cnn = model
        elif self.model_name == 'resnet50':    
            model = models.resnet50(pretrained = pretrained )
            model.fc= nn.Identity()
            self.cnn = model
        elif self.model_name == 'resnet18':    
            model = models.resnet18(pretrained = pretrained )
            model.fc= nn.Identity()
            self.cnn = model
        else :
            raise ValueError(f"Unsupported CNN backbone type . Supported models: 'resnet18', 'resnet50', 'mobilenet_v2', 'inception_v3'")
        if not trainable :
            for name, param in self.cnn.named_parameters():
                param.requires_grad = False  # Freeze layer
        
        x = torch.randn(1, 3,224,224)
        self.feature_size = self.cnn(x).shape[-1] 
        
        self.projection = nn.Sequential(
            nn.Dropout(p=0.2) , 
            nn.Linear(self.feature_size , self.embed_size) , 
            nn.BatchNorm1d( self.embed_size ) , # N batch_size
            nn.ReLU()
        )
    def forward(self, images):
        """
        Forward pass to extract features from images.
        
        Args:
            images (torch.Tensor): Batch of input images [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Image features [batch_size, embed_size]
        """
        # Extract features from CNN

        patches = create_overlapping_grid(images , self.grid_size , self.overlap_ratio)
        
        features = self.cnn(patches)
        
        if self.project:
            # Project features to the specified embedding size
            features = self.projection(features)
        
        
        features = features.reshape(images.shape[0] , self.grid_size **2 , -1  )
        
        return features
    
    def get_feature_size(self):
        """Returns the raw feature size of the CNN backbone"""
        return self.feature_size




def get_kernel( input_size , grid_size , overlap_ratio ):
  kernel_size = int(input_size // ( ( grid_size-1 )* (1-overlap_ratio) + 1 ))
  stride =  int( kernel_size * (1-overlap_ratio))
  return kernel_size , stride 


def create_overlapping_grid(image_tensor, grid_size, overlap_ratio):
    """
    Crops a single image tensor [N , C, H, W] into an overlapping grid
    of sub-images using tensor operations without explicit iteration over batches.

    Args:
        image_tensor (torch.Tensor): The input image with shape [C, H, W].
        grid_size (int): The number of rows and columns in the grid.
        overlap_ratio (float): The ratio of overlap between adjacent sub-images (0.0 to 1.0).

    Returns:
        torch.Tensor: A tensor containing the sub-images, reshaped into
                      [grid_size * grid_size, C, sub_h, sub_w].
    """
    
    b, c, h, w = image_tensor.shape

    kernel_size , stride = get_kernel(w , grid_size , overlap_ratio )

    all_patches = []
    for i in range(grid_size):
      for j in range(grid_size):
        top = i * stride
        left = j * stride
        bottom = top + kernel_size
        right = left + kernel_size
        sub_image = image_tensor[:,:, top:bottom, left:right]
        
        all_patches.append(sub_image)
    
    stacked_patches = torch.stack(all_patches, dim=0) # Shape: [grid_size*grid_size, B, C, sub_h, sub_w]

    # Permute dimensions to bring the batch dimension before the patch index dimension
    permuted_patches = stacked_patches.permute(1, 0, 2, 3, 4) # Shape: [B, grid_size*grid_size, C, sub_h, sub_w]

    # Reshape to have batch * num_patches as the first dimension
    final_patches = permuted_patches.reshape(b * grid_size * grid_size, c, kernel_size, kernel_size)

    return final_patches


