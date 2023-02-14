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

class myDense(nn.Module):
    def __init__(self, size):
        super(myDense, self).__init__()
        self.hidden = nn.ModuleList()
        n = 1024
        i = 0
        while i < size and n > 204:
            self.hidden.append(nn.Linear(n, int(int(n)/2)))
            i = i + 1
            n = n/2
        while i < size:
            self.hidden.append(nn.Linear(int(n), int(n)))
            i = i + 1
        self.output = nn.Linear(n, 102)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.log_softmax = nn.LogSoftmax(dim = 1)
    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.output(x)
        x = self.log_softmax(x)
        return x
                      
        

def dense(hidden_units):
    # TODO: Build and train your network
    #Freeze pretrained model parameters
    model = models.densenet121(pretrained = True)
    #print(model)
    for param in model.parameters():
        param.requires_grad = False
    #Define classifier (Note: May need to change size of layers to match flattened image)
    
    #TODO: Possibly create a class for the classifier? Need a way to create as many hidden units as the user wants
    if hidden_units != None: 
        classifier = myDense(hidden_units)
    else:
        classifier = nn.Sequential(nn.Linear(25088, 1000), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(1000, 200), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(200, 102), nn.LogSoftmax(dim = 1))
    #Attach classifier to model
    
    model.classifier = classifier
    #Train and validate the model

    #If Adam doesn't work, change back to SGD
    #print("End of dense function")
    return model

def dense_load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained = True)
    #print(model)
    for param in model.parameters():
        param.requires_grad = False
    #Define classifier (Note: May need to change size of layers to match flattened image)
    hidden = checkpoint['hidden_units']
    
    classifier = myDense(hidden)
    #Attach classifier to model
    model.classifier = classifier
    optimizer = optim.SGD(model.classifier.parameters(), checkpoint['learn_rate'])
    
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
