from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch
from enum import Enum
import torch.nn.functional as F
from workspace_utils import active_session
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from vgg import *

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    #print(type(image))
    image.thumbnail((256,256))
    #print(type(image))
    width, height = image.size
    image = image.crop(((width-224)/2, (height-224)/2, (width + 224)/2, (height + 224)/2))

    #print(type(image))
    #print(image)
    np_image = np.array(image)
    #print(type(np_image))
    #print(np_image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # print(np_image[0])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose(2, 0, 1)
    #print(np_image.shape)
    return np_image

#image = Image.open(r"/home/workspace/aipnd-project/flowers/train/1/image_06734.jpg")
#image
#print(type(image))
#np_im = process_image(image)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
#tensor_im = torch.from_numpy(np_im)
#imshow(tensor_im)

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu == True:
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()
    image = Image.open(image_path)
    np_im = process_image(image)
    tensor_im = torch.from_numpy(np_im)
    tensor_im = tensor_im.view(1, 3, 224, 224).float()
    #print(tensor_im.shape)
    with torch.no_grad():
        #Move stuff to cuda if specified
        tensor_im = tensor_im.to(device)
        output = model(tensor_im)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk)
    class_to_idx_dict = model.class_to_idx
    idx_to_class_dict = dict(map(reversed, class_to_idx_dict.items()))
    top_class_list = []
    for i in range(topk):
        my_tensor = top_class[0,i]
        #print(my_tensor)
        #print(my_tensor.shape)
        integer = my_tensor.item()
        top_class_list.append(idx_to_class_dict[integer])
    return top_p, top_class_list