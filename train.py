# importing all the libraries

import os
import shutil

import torch
import torchvision
import torch.nn

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import random_split


import PIL 
from PIL import Image
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchinfo import summary
import yaml



# configuration for the encoders
config_path_encoder = "./config.yaml"

with open(config_path_encoder, "r") as stream:
    try:
        encoder_configuration = yaml.safe_load(stream=stream)
        encoder_configuration = encoder_configuration.get("encoder")
    except yaml.YAMLError as exc:
        print(exc)



# image transformation to required mean and std

tfms_content = T.Compose([
    T.Normalize(encoder_configuration.get("_stats_content")),
    T.RandomCrop(encoder_configuration.get("random_crop_dimension"))
])
train_tfms = T.Compose([
    T.Normalize(encoder_configuration.get("_stats")),
    T.RandomCrop(encoder_configuration.get("random_crop_dimension")),
])

test_tfms = T.Compose([
    T.Normalize(encoder_configuration.get("_stats")),
    T.RandomCrop(encoder_configuration.get("random_crop_dimension")),
])


# Data loading and preprocessing train
content_dataset = ImageFolder('./dataset/img', tfms_content)
style_dataset = ImageFolder('./dataset/style', train_tfms)
colored_dataset = ImageFolder('./dataset/mask', train_tfms)

# Data loading and preprocessing test
content_dataset_test = ImageFolder('./dataset/img', test_tfms)
style_dataset_test = ImageFolder('./dataset/style', test_tfms)
colored_dataset_test = ImageFolder('./dataset/mask', test_tfms)

# Initializing the dataloader train
content_dl = DataLoader(content_dataset, batch_size = encoder_configuration.get("batch_size"), shuffle = True, num_workers = 2, drop_last = True)
style_dl = DataLoader(style_dataset, batch_size = encoder_configuration.get("batch_size"), shuffle = True, num_workers = 2, drop_last = True)
colored_dl = DataLoader(colored_dataset, batch_size = encoder_configuration.get("batch_size"), shuffle = True, num_workers = 2, drop_last = True)

# Initializing the dataloader test
content_dl_test = DataLoader(content_dataset_test, batch_size = 1, shuffle = True, num_workers = 2, drop_last = True)
style_dl_test = DataLoader(style_dataset_test, batch_size = 1, shuffle = True, num_workers = 2, drop_last = True)
colored_dl_test = DataLoader(colored_dataset_test, batch_size = 1, shuffle = True, num_workers = 2, drop_last = True)




# Device and Data Loader Utility Functions for Model Training

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def to_device(data, device):
    # takes in the data and converts to the device tensors
    if isinstance(data, (list,tuple)):
        return [to_device(x, device=device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    

device = get_default_device()
print("Device found",device)
print("\n\n\n")


# loading the dataloader train to device

content_dl = DeviceDataLoader(content_dl, device)
style_dl = DeviceDataLoader(style_dl, device)
colored_dl = DeviceDataLoader(colored_dl, device)

# loading the dataloader test to device

content_dl_test = DeviceDataLoader(content_dl_test, device)
style_dl_test = DeviceDataLoader(style_dl_test, device)
colored_dl_test = DeviceDataLoader(colored_dl_test, device)


# Model pipeline

# The encoder is used for all the three kinds of image set used

vg19 = vgg19(True)
print(vg19)
print("\n\n\n")


# Encoder model with specified layers to be used

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the vgg layers to be used here

        vgg = vgg19(weights = 'DEFAULT').features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]

        # we don't the parameters to be updated so set the upgrade_grad = False
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_middle_feature_only=False):
        # Here the images are loaded from device and do we need the last ourput features or not

        """ Since we know that we have 
            black and white image = middle as well deep layers for(style and content loss)
            style image = middle layer features for adain
            colored_image = middle layer features for adain and style loss
                combined with the style image 

            Args : need images and the weights

            returns the encodings computed
        """

        # Pass the input image through the features

        h1 = self.slice1(images)    
        h2 = self.slice2(h1)    
        h3 = self.slice3(h2)    
        h4 = self.slice4(h3)   

        if output_middle_feature_only is True:
            return h1, h2, h3
        else:
            return h4
        

# Object of the encoder model 

enc = VGGEncoder()
summary(enc, input_size=(encoder_configuration.get("batch_size"),3, 512, 512))
         
        

""" 
    ADAIN 
    ADAIN will be used here over the middle and initial encodings of
    the style and colored images
    Calculate the mean and std of middle and inital layers of style image
    and colored image
    Normalize the colored image and shift them using the std and mean of
    the style image
"""

def calc_mean_std(features):
    """
        ARGS : Tensor (Input feature) (bs , channel , w, h)
        returns : mean and std for every layer 
        mean = (bs, channel , 1, 1)
        std = (bs, channel, 1, 1)
    """

    # Get the batch size and output channels 

    batch_size, c = features.size()[:2]

    # Calculate the mean and reshape it to match the required shape

    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)

    # Calculate the std and reshape it to match the required shape

    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1)+1e-6

    return features_mean, features_std

def ADAIN(colored_features, style_features):
    """
        colored_features : middle and initial layer features of colored image (bs, channel ,w , h) 
        style_features : middle and initial layer features of style image (bs, channel ,w , h)
        
        ARGS: features
        
        Returns: shifted feature maps of colored image using style image
    """

    colored_features_mean, colored_features_std = calc_mean_std(colored_features)
    style_features_mean, style_features_std = calc_mean_std(style_features)

    # Normalizing the colored features 

    normalized_features = style_features_std*(colored_features-colored_features_mean)/colored_features_std + style_features_mean

    return normalized_features




class RC(torch.nn.Module):
    """
        A wrapper for reflection and Conv2d
        
        Args: 
            in_channles (int): Number of channels
            out_channels (int): Number of channels
            kernel_size (int): size of the convolution filters
            pad_size (int): padding
            activated (int): whether to apply activation function
    """

    def __init__(self, in_channels, out_channels, kernel_size = 3, pad_size = 1, activated = True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        """
        Forward pass of RC
        
        ARGS : input of size (bs, channel, height and widht)
        
        return : size (bs, out_channels, height and width)"""

        h = self.pad(x)
        h = self.conv(h)
        if self.activated is True:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        """
            Forward pass of the decoder

            ARGS : features deep layer from the encoder of content image

            Return : Generated image
        """

        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)      # Perform another upsampling using F.interpolate with scale factor 2
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)      # Perform another upsampling using F.interpolate with scale factor 2
        h = self.rc8(h)
        h = self.rc9(h)
        return h
    
dec = Decoder()


def gram_matrix(features):
        batch_size,c,h,w=features.size()
        features=features.view(batch_size,c,-1)
        gram_cal=torch.bmm(features,features.transpose(1,2))
        return gram_cal/(c*h*w)

class Model(nn.Module):

    def __init__(self, enc):
        super().__init__()
        self.vggencoder = VGGEncoder()
        self.decoder = Decoder()
    """ 
        Loss computing and image generation 
        For loss computing we will need 2 types of losses for unsupervised
        Content loss
        Style loss

        Content loss = calculated between deep features of the content and generated image
        Style loss = calculated between middle features of content and 

        ARGS:
            content_image (torch.tensor)
            style_image (torch.tensor)
            colored_image (torch.tensor)

        returns stylized output image
    """
    def generate(self, content_images):
        """
            The function helps to generate the image 

            ARGS : content images to convert to colored images
            returns : colored image
        """

        content_features = self.vggencoder(content_images)
        out = self.decoder(content_features)
        return out
   
    @staticmethod
    def calc_content_loss(generated_features, content_features):
        """
            Calculate the content loss bw the generated image and content image
            through features of their deep network

            ARGS : torch.tensor() : Output features of the generated image and output
                    features of the content image

            return : content loss
        """
        return F.mse_loss(content_features, generated_features)
    @staticmethod
    def calc_style_loss(generated_middle_features, style_middle_features, colored_middle_features):
        gram_generated = [gram_matrix(f) for f in generated_middle_features]
        gram_style = [gram_matrix(f) for f in style_middle_features]
        gram_colored = [gram_matrix(f) for f in colored_middle_features]
        loss = 0
        for gen, style, color in zip(gram_generated, gram_style, gram_colored):
            loss += F.mse_loss(gen, style) + F.mse_loss(gen, color)
        return loss / len(generated_middle_features)

    def forward(self, content_images, style_images, colored_images, alpha = 0.5):
        """
            Forward pass of the model

            ARGS: 
                content_images (torch.Tensor): Input content images as tensors.
                style_images (torch.Tensor): Input style images as tensors.  
                alpha (float, optional): Style strength factor. Default is 1.0.
                lam (float, optional): Weight of the style loss. Default is 10.

            return : total loss
        """         

        # content loss
        content_features = self.vggencoder(content_images, output_middle_features_only = False) # get the deep layers
        output = self.decoder(content_features)
        generated_features = self.vggencoder(output, output_middle_features_only = False)
        content_loss = self.calc_content_loss(generated_features=generated_features, content_features=content_features)

        # style loss
        style_middle_features = self.vggencoder(style_images, output_middle_features_only = True)
        colored_middle_features = self.vggencoder(colored_images, output_middle_features_only = True)
        generated_middle_features = self.vggencoder(output, output_middle_features_only = True)
        style_loss = self.calc_style_loss(generated_middle_features=generated_middle_features,          style_middle_features=style_middle_features,colored_middle_features=colored_middle_features)

        total_loss = content_loss*alpha + (1-alpha)*style_loss
        return total_loss

def denorm(tensor, device):
    """
    Denormalizes the image tensor using the mean and standard deviation values of ImageNet.

    Args:
        tensor (torch.Tensor): The input tensor to be denormalized. It should have shape (C, H, W).
        device (str or torch.device): The device on which the computations should be performed.

    Returns:
        torch.Tensor: The denormalized tensor with values clamped between 0 and 1.
    """
    # Define the standard deviation values for each channel (R, G, B)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)

    # Define the mean values for each channel (R, G, B)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)

    # Perform denormalization and clamp the tensors between value of 0 and 1 by applying the following formula:
    # Denorm = (Input * STD) + Mean
    denormalized_tensor = torch.clamp(tensor * std + mean, 0, 1)

    return denormalized_tensor

def main(): 
    enc=VGGEncoder()
    model=Model(enc)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5  # Change as required
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for content, style, color in zip(content_dl, style_dl, colored_dl):
            content, style, color = to_device(content[0], device), to_device(style[0], device), to_device(color[0], device)

            optimizer.zero_grad()
            loss = model(content, style, color, alpha=0.5)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for content, style, color in zip(content_dl_test, style_dl_test, colored_dl_test):
            content, style, color = to_device(content[0], device), to_device(style[0], device), to_device(color[0], device)
            output = model.generate(content)
            plt.imshow(denorm(output[0].cpu(), device).permute(1, 2, 0))
            plt.show()
            break  # Show one example

if __name__=="__main__":
    main()