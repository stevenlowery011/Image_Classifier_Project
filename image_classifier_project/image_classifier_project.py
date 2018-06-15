"""
image_classifier_project.py
This module contains functions and classes relating to the neural network classifier.
This module is referenced by train.py and predict.py
Classes:
    Network(nn.Module) - Builds a feedforward network with arbitrary hidden layers
Functions:
    get_model() - Parse the given model name and create a new transfer model
    model_config() - 
    validation() - 
    network_train() - 
    save_checkpoint() - 
    load_checkpoint() - 
    process_image() - 
    imshow() - 
    predict() - 
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
    Parse the given model name and create a new transfer model.
    Parameters:
        model_name - string, name of the transfer model from torchvision.models
    Returns:
        model - torchvision.models.model, the transfer network model with gradients disabled
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
    
    Parameters:
        data_dir - string, the data directory
        model - torchvision.models.model, the neural network model
        hidden_layers - list, the hidden layers of the classifier
        dropout_probability - float, the probability of dropout during model training
    Returns:
        model - torchvision.models.model - the output model with the classifier attached
        train_dataloader - the dataloader for the training set
        test_dataloader - the dataloader for the test set
        total_steps - the total number of iterations in the training dataset
    """
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'

    # Create the transforms for the image data
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

    # Get the datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = True)

    # Check the classifier of the given model and find the first Linear module to see how many inputs it takes
    classifier_modules = [type(model.classifier[i]).__name__ for i in range(len(model.classifier))]
    First_Linear_Module = classifier_modules.index('Linear')

    # Create the classifier for the model 
    # Use the transfer models outputs as inputs and the number of image categories as outputs
    classifier = Network(model.classifier[First_Linear_Module].in_features, len(train_dataset.classes), hidden_layers, drop_p=dropout_probability)
    
    # Attach the classifier to the model
    model.classifier = classifier

    # Calculate how many steps it will take to iterate through all the training images
    total_steps = ceil(len(train_dataset.samples)/train_dataloader.batch_size)

    return model, train_dataloader, test_dataloader, total_steps


def validation(model, testloader, criterion, processor):
    """
    Validates the model using the test images.

    Parameters:
        model - torchvision.models.model, the neural network model
        testloader - the dataloader for the test set
        criterion - the output loss criterion
        processor - torch.device, CPU of GPU
    Returns:
        test_loss - torch.tensor, the output loss from the validation pass
        accuracy - torch.tensor, the percentage of correctly classified test images
    """
    test_loss = 0
    accuracy = 0

    # Send the model to the desired processor
    model.to(processor)

    # Iterate through the test dataloader and do a validation pass on all images
    for images, labels in testloader:
        images, labels = images.to(processor), labels.to(processor)

        # Do a forward pass
        output = model.forward(images)

        # Calculate the loss based on the criterion
        test_loss += criterion(output, labels).item()

        # Since the output is log-loss, the probabilities are the e^x of the output
        ps = torch.exp(output)

        # Check how many of the most probable output results match the labels
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def network_train(model, epochs, learnrate, train_dataloader, test_dataloader, total_steps, use_GPU):
    """
    Train a neural network model and update the output window with output loss, accuracy, and time.

    Parameters:
        model - torchvision.models.model, the neural network model
        epochs - int, number of training epochs (iterations)
        learnrate - the neural network learning rate
        train_dataloader - the dataloader for the training set
        test_dataloader - the dataloader for the test set
        total_steps - int, the total number of iterations in the training dataset
        use_GPU - bool, use GPU if True
    Returns:
        accuracy - float, the accuracy of the model
    """
    print("\nTraining network...")
    
    # Select the GPU if it is available and if use_GPU is True
    if(use_GPU==True):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU not available. Using CPU instead.")
            device = torch.device("cpu")
    # Otherwise use the CPU
    else:
        device = torch.device("cpu")
    print(f"Device = {device}\n")
    
    # Set up the training parameters
    steps = 0
    running_loss = 0
    print_every = total_steps

    # Define the criterion and optimizer given the model and learning rate
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)

    # Send the model to the correct processor
    model.to(device)

    # Record the start time before training
    start = time.time()
    
    # Train the network all the way through the training set for each epoch
    for e in range(epochs):
        steps = 0

        # Put the model in training mode
        model.train()

        # Loop through the training dataloader and train on all images
        for image, label in train_dataloader:
            steps += 1
            # If the dataloader is too long to process, the number of steps can be reduced and break mid-training
            if steps > total_steps:
                break

            # Print an update to the output window to show that the training is ongoing
            print("Epoch: {:^3}/{:^3}.. ".format(e+1, epochs),
                  "Step: {:^3}/{:^3}..".format(steps, total_steps),
                  "-- Time: {:>10.3f}  ".format(time.time()-start),
                  "(Model is training. Validation after {} more training steps.)".format(print_every - steps))

            # Send the data to the processor
            image, label = image.to(device), label.to(device)

            # Zero out the gradients before doing forward pass and back-propagation
            optimizer.zero_grad()

            # Do a forward pass, calculate loss, and do a back-propagation
            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward()

            # Optimize the weights and biases
            optimizer.step()

            # Update the loss value
            running_loss += loss.item()

            # Do a validation pass once every so many loops
            if steps % print_every == 0:
                # Put the model in eval mode and turn off gradients
                model.eval()
                with torch.no_grad():
                    # Do a validation
                    test_loss, accuracy = validation(model, test_dataloader, criterion, device)
                # Print the results to the output window
                print("Epoch: {:^3}/{:^3}.. ".format(e+1, epochs),
                      "Step: {:^3}/{:^3}..".format(steps, total_steps),
                      "-- Time: {:>10.3f}".format(time.time()-start),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)),
                      )
                # Clear the running loss and put the model back in training mode
                running_loss = 0
                model.train()

    # Return the accuracy in terms of a percentage
    return accuracy/len(test_dataloader)


def save_checkpoint(filepath, model, transfer_model_name, input_size, output_size, hyperparams, model_accuracy):
    """
    Save a model checkpoint.

    Parameters:
        filepath - string, the filepath of the checkpoint to be saved
        model - torchvision.models.model, the trained model to be saved
        transfer_model_name - string, the name of the transfer model 
        input_size - int, the input size of the classifier's hidden layers
        output_size - int, the output size of the classifier (number of categories)
        hyperparams - dictionary, the hyperparameters of the model training, including epochs, dropout probability, and learnrate
        model_accuracy - float, the accuracy of the trained model
    Returns:
        None
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
    """
    Load a model checkpoint.

    Parameters:
        filepath - string, the path to the saved checkpoint to be loaded
    Returns:
        model - the trained model from the checkpoint file
        accuracy - float, the accuracy of the trained model
    """
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
