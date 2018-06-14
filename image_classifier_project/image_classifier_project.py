"""
image_classifier_project.py docstring
"""

import argparse
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import time
from workspace_utils import active_session
import os
from PIL import Image
from math import ceil

def main():
    print("In main(). This script contains functions and classes relating to train.py and predict.py")

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


def get_model(model_name):
    """
    docstring
    """
    if model_name=='vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name=='vgg13':
        model = models.vgg13(pretrained=True)
    elif model_name=='vgg11':
        model = models.vgg11(pretrained=True)
    elif model_name=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("\nINPUT ERROR: must select from one of these models: vgg16, vgg13, vgg11, alexnet\n")

    for param in model.parameters():
        param.requires_grad = False

    return model


def model_config(data_dir, model, hidden_layers, dropout_probability):
    """
    docstring
    """
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = True)

    classifier_modules = [type(model.classifier[i]).__name__ for i in range(len(model.classifier))]
    First_Linear_Module = classifier_modules.index('Linear')
    classifier = Network(model.classifier[First_Linear_Module].in_features, len(train_dataset.classes), hidden_layers, drop_p=dropout_probability)
    model.classifier = classifier

    total_steps = ceil(len(train_dataset.samples)/train_dataloader.batch_size)

    return model, train_dataloader, test_dataloader, total_steps


def validation(model, testloader, criterion, processor):
    """
    docstring
    """
    test_loss = 0
    accuracy = 0
    model.to(processor)
    for images, labels in testloader:
        images, labels = images.to(processor), labels.to(processor)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def network_train(model, epochs, learnrate, train_dataloader, test_dataloader, total_steps, use_GPU):
    """
    docstring
    """
    print("\nTraining network...")
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if(use_GPU==True):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU not available. Using CPU instead.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Device = {device}\n")
    
    steps = 0
    running_loss = 0
    print_every = total_steps
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    model.to(device)
    start = time.time()
    
    for e in range(epochs):
        steps = 0
        model.train()
        for image, label in train_dataloader:
            steps += 1
            if steps > total_steps:
                break
            print("Epoch: {:^3}/{:^3}.. ".format(e+1, epochs),
                  "Step: {:^3}/{:^3}..".format(steps, total_steps),
                  "-- Time: {:>10.3f}  ".format(time.time()-start),
                  "(Model is training. Validation after {} more training steps.)".format(print_every - steps))
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, test_dataloader, criterion, device)
                print("Epoch: {:^3}/{:^3}.. ".format(e+1, epochs),
                      "Step: {:^3}/{:^3}..".format(steps, total_steps),
                      "-- Time: {:>10.3f}".format(time.time()-start),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)),
                      )
                running_loss = 0
                model.train()

    return accuracy/len(test_dataloader)


def save_checkpoint(filepath, model, transfer_model_name, input_size, output_size, hyperparams, model_accuracy):
    """
    docstring
    """
    checkpoint = {'transfer_model_name': transfer_model_name,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hyperparams['hidden_layers'],
                  'drop_p': hyperparams['dropout_probability'],
                  'learnrate': hyperparams['learnrate'],
                  'epochs': hyperparams['epochs'],
                  'model_accuracy': model_accuracy,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_name = checkpoint['transfer_model_name']
    model = get_model(model_name)
    
    classifier = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'],
                         checkpoint['drop_p'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    hidden_layers = checkpoint['hidden_layers']
    dropout_probability = checkpoint['drop_p']
    learnrate = checkpoint['learnrate']
    epochs = checkpoint['epochs']
    accuracy = checkpoint['model_accuracy']
    
    return model, accuracy


def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model
    
    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(pil_image)
    
    # add dimension for batch
    img_tensor.unsqueeze_(0)
    
    return img_tensor


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


def predict(image_path, model, use_GPU, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    image = process_image(Image.open(image_path))
    
    if(use_GPU==True):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU not available. Using CPU instead.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Device = {device}\n")

    # do a forward pass on the image
    model.eval()
    model.to(device)
    criterion = nn.NLLLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)

    topk_probs_tensor, topk_idx_tensor = ps.topk(topk)

    #probs = topk_probs_tensor.tolist()[0]
    #classes = [model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (topk_idx_tensor).tolist()[0]]
    
    return topk_probs_tensor, topk_idx_tensor #probs, classes


if __name__ == '__main__':
    main()
