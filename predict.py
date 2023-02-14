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
from dense import *
from predict_utils import *

#TODO: How to make some arguments required? Or create default values for some options?
parser = argparse.ArgumentParser()
parser.add_argument('imagefile', metavar = 'imagefile', type = str, help = 'The filepath to the image you want to classify')
parser.add_argument('checkpointfile', type = str, help = 'The filepath to the model checkpoint you want to use for inference. If checkpoint is from a vgg16 model, filename must be vggcheckpoint.pth. If checkpoint is from a densenet121 model, filename must be densecheckpoint.pth.')
parser.add_argument('--top_k', type = int, default = 1, help = 'The number of top classes the program should output. Defaults to 1.')
parser.add_argument('--category_names', type = str, default = '/home/workspace/ImageClassifier/cat_to_name.json', help = 'The file to load flower names from. Defaults to /home/workspace/ImageClassifier/cat_to_name.json')
parser.add_argument('--gpu', action = 'store_true', help = 'Use gpu for training. Gpu will be used if this flag is added')
args = parser.parse_args()

#First up: Load the checkpoint!

if "vgg" in args.checkpointfile:
    model = vgg_load_checkpoint(args.checkpointfile)
elif "dense" in args.checkpointfile:
    model = dense_load_checkpoint(args.checkpointfile)
    
    
#Next: Get json file for cat to name
    
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
#print(cat_to_name)


#Open the image and pre-process it 

image = Image.open(args.imagefile)
#image
#print(type(image))
#np_im = process_image(image)
#tensor_im = torch.from_numpy(np_im)
#imshow(tensor_im)
# TODO: display label for input image
# print(cat_to_name['1'])

#fig, ax = plt.subplots()
top_p, top_class = predict(args.imagefile, model, gpu = args.gpu, topk=args.top_k)
top_class_names = []
for i in range(args.top_k):
    top_class_names.append(cat_to_name[top_class[i]])
#print(top_p[0])
for i in range(args.top_k):
    print(f"{i+1}: {top_class_names[i]}")
    print(f"Probability: {top_p[0,i].item()}")
