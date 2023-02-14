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
#How to make certain arguments required?
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', metavar = 'data_dir', type = str)
parser.add_argument('--save-dir', type = str, nargs = '?', default = '/home/workspace/saved_models', help = 'Directory to save checkpoints to. Defaults to /home/workspace/saved_models.')
parser.add_argument('--arch', type = str, nargs = '?', default = 'vgg', help = 'Model architecture. Enter vgg (for vgg 16) or dense (for densenet). Default is vgg.')
parser.add_argument('--learning_rate', type = float, nargs = '?', default = 0.01, help = 'Learning rate for the model.')
parser.add_argument('--hidden_units', type = int, nargs = '?', help = 'Number of hidden units for the model', default = 5)
parser.add_argument('--epochs', type = int, nargs = '?', default = 5)
parser.add_argument('--gpu', action = 'store_true', help = 'Use gpu for training. Gpu will be used if this flag is added')
args = parser.parse_args()
#print(args.data_dir)
#print(args.save_dir)
#print(args.gpu)

# Arguments are all working! Access them with args.var_name

#Step 1: Load the data
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# data_transforms = 
training_transforms = transforms.Compose([transforms.Resize((256,256)),
    transforms.RandomCrop(224),
                               transforms.RandomHorizontalFlip(p=0.5), 
                                         transforms.Resize((224,224)), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                    transforms.Normalize((0.485,0.456, 0.406), (0.229, 0.224, 0.225))])
    
testing_transforms = transforms.Compose([transforms.Resize((256,256)),
    transforms.Resize((224, 224)), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# TODO: Load the datasets with ImageFolder
# image_datasets = 
trainset = datasets.ImageFolder(train_dir, transform = training_transforms)
# print(len(trainset.classes))
testset = datasets.ImageFolder(test_dir, transform = testing_transforms)
validset = datasets.ImageFolder(valid_dir, transform = testing_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
# dataloaders = 
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64)
validloader = torch.utils.data.DataLoader(validset, batch_size = 64)

# Data loaded!

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
# print(cat_to_name)

# JSON file (cat_to_name) loaded!

#Model needs to be of type arch and have hidden_units hidden layers

# Make a function for vgg arch
# Give usage for arch (vgg or DenseNet, etc.)
if args.arch == 'vgg':
    #print("Hidden units fixed. User hidden units not implemented yet.")
    model = vgg(args.hidden_units)
    #print(model)
elif args.arch == 'dense':
    #print("Not implemented yet. Stay tuned!")
    model = dense(args.hidden_units)
    #print(model)

print("Created model")
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate) #Change lr back to 0.1 if 0.2 or 0.15 doesn't
print("Training")
# Train the model
# 3 hidden units works well (?)
if args.gpu == True:
    device = 'cuda'
else:
    device = 'cpu'
model.to(device) # move model to GPU for training
epochs = args.epochs
with active_session(): # long running work
    for e in range(epochs):
        print("Epoch: ", e)
        for batchnum, (images, labels) in enumerate(trainloader):
            # Train
            # print(images.shape)
            #Flatten images (currently not in use)
            # images = images.view(images.shape[0], -1)
            # print(images.shape)
            #Set gradients to zero at beginning of each batch
            optimizer.zero_grad()
            #Forward pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #Calculate loss
            loss = criterion(outputs, labels)
            print("Training Loss: ", loss)
            #Backward pass with gradient descent
            loss.backward()
            #Update weights with optimizer
            optimizer.step()
            # print("One batch complete!")
        # print("One epoch complete!")
        else:
            # Validation time!
            print("Validating...")
            with torch.no_grad():
                model.eval()
                for batchnum, (images, labels) in enumerate(validloader):
                    # images = images.view(images.shape[0], -1) # Flatten the images
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images) # NOTE: added this line since porting from Jupyter
                    loss = criterion(outputs, labels) # NOTE: added this line since porting from Jupyter
                    print("Validation Loss: ", loss) # NOTE: added this line since porting from Jupyter
                    ps = torch.exp(model(images))
                    top_probability, top_class = ps.topk(1, dim=1) # gets top class for each image w/ placeholder var for top prob
                    equals = top_class == labels.view(*top_class.shape) # boolean for whether each image was classified correctly
                    
                    accuracy = torch.mean(equals.type(torch.FloatTensor)) # takes mean of boolean array to get overall accuracy
                    print(f"Accuracy: {accuracy.item()*100} %")
                model.train()

#After training: Save checkpoint

model.class_to_idx = trainset.class_to_idx
#TODO: Change input and output size to reflect user hidden layers
if args.arch == 'vgg':
    checkpoint = {'input_size' : 25088, 'output_size' : 102,
                 'epochs' : args.epochs, 'optim_state' : optimizer.state_dict, 'state_dict' : model.state_dict(), 'class_to_idx' : model.class_to_idx, 'hidden_units': args.hidden_units, 'learn_rate' : args.learning_rate}
    torch.save(checkpoint, args.save_dir + '/vggcheckpoint.pth') # Changed to accomodate command line args

elif args.arch == 'dense':
    checkpoint = {'input_size' : 1024, 'output_size' : 102,
                 'epochs' : args.epochs, 'optim_state' : optimizer.state_dict, 'state_dict' : model.state_dict(), 'class_to_idx' : model.class_to_idx, 'hidden_units': args.hidden_units, 'learn_rate' : args.learning_rate}
    torch.save(checkpoint, args.save_dir + '/densecheckpoint.pth')

    