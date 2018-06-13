"""
train.py docstring
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
from image_classifier_project import *


def main():
    in_arg = get_input_args()
    print(in_arg)
    model = get_model(in_arg.arch)
    model, train_dataloader, test_dataloader, total_steps = model_config(in_arg.data_dir, model, in_arg.hidden_units, in_arg.drop_p)
    print(model)
    print(train_dataloader)
    print(total_steps)
    network_train(model, in_arg.epochs, in_arg.learning_rate, train_dataloader, test_dataloader, total_steps)



def get_input_args():
    """
    docstring
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store", type=str,
                        help='data directory of training images')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints/',
                        help='directory to save model checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_units', action="append", type=int, default=[],
                        help='hidden layer units')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--drop_p', type=float, default=0.2,
                        help='dropout probability')
    return parser.parse_args()

def get_model(model_name):
    if model_name=='vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("\nINPUT ERROR: must select from one of these models: vgg16\n")

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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle = True)

    classifier = Network(model.classifier[0].in_features, len(train_dataset.classes), hidden_layers, drop_p=dropout_probability)
    model.classifier = classifier

    #total_steps = sum(1 for e in enumerate(train_dataloader))
    total_steps = 103

    return model, train_dataloader, test_dataloader, total_steps


def validation(model, testloader, criterion, processor):
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


def network_train(model, epochs, learnrate, train_dataloader, test_dataloader, total_steps):
    print("Training network...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device = {device}")
    
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
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Step: {}/{}..".format(steps, total_steps))
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
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Step: {}/{}..".format(steps, total_steps),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)),
                      "-- Time: {:.3f}".format(time.time()-start))
                running_loss = 0
                model.train()


if __name__ == '__main__':
    main()
