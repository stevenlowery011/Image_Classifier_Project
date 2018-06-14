"""
train.py
This script takes a data directory of categorized images
and trains a neural network based on transfer characteristics
of a pre-trained neural network, then saves the newly trained
model. The model will be trained to predict the category of
the input image.
"""

import argparse
from image_classifier_project import *


def main():
    """
    Main function for train.py - parses command line arguments for data directory,
    save directory, checkpoint name, transfer model architecture, learning rate,
    classifier network hidden units dimensions, number of epochs, dropout probability,
    and whether or not to use the GPU for training. 
    Prints the training information to the output window then begins training.
    During model training, the number of steps is printed after every training step.
    A validation step is done once after each epoch, printing the training loss, test
    loss, and model accuracy. 
    The model is then saved along with the model accuracy and relevant hyperparameters.
    """
    in_arg = get_input_args()

    # Set the default hyperparameters if none given
    if in_arg.hidden_units==[]:
        in_arg.hidden_units=[128, 64, 32];

    # Create the save file path name
    save_path = in_arg.save_dir + in_arg.checkpoint

    # Organize the inputs into the hyperparameters dictionary
    hyperparameters = {'architecture': in_arg.arch,
                       'hidden_layers': in_arg.hidden_units,
                       'dropout_probability': in_arg.drop_p,
                       'learnrate': in_arg.learning_rate,
                       'epochs': in_arg.epochs}

    # Get the transfer model
    model = get_model(hyperparameters['architecture'])

    # Create the dataloaders for training and testing images
    # Also, create the classifier based on the given inputs and attach it to the transfer model
    model, train_dataloader, test_dataloader, total_steps = model_config(in_arg.data_dir, model, 
                                                                         hyperparameters['hidden_layers'], 
                                                                         hyperparameters['dropout_probability'])

    # Print the relevant parameters to the output window before training
    print("\n")
    print("Transfer model:               {}".format(hyperparameters['architecture']))
    print("Hidden layers:                {}".format(hyperparameters['hidden_layers']))
    print("Learning rate:                {}".format(hyperparameters['learnrate']))
    print("Dropout probability:          {}".format(hyperparameters['dropout_probability']))
    print("Epochs:                       {}".format(hyperparameters['epochs']))
    print("Training data:                {}".format(in_arg.data_dir + '/train'))
    print("Validation data:              {}".format(in_arg.data_dir + '/test'))
    print("Checkpoint will be saved to:  {}".format(save_path))

    # Start training the network and return the model accuracy after training
    model_accuracy = network_train(model, 
                                   hyperparameters['epochs'], 
                                   hyperparameters['learnrate'], 
                                   train_dataloader, test_dataloader, 
                                   total_steps, in_arg.gpu)

    # Save the model checkpoint along with the transfer model architecture it came from and the model's trained accuracy
    save_checkpoint(save_path, model, hyperparameters['architecture'], 
                    model.classifier.hidden_layers[0].in_features, 
                    model.classifier.output.out_features, 
                    hyperparameters, model_accuracy)


def get_input_args():
    """
    Parse command line arguments.

    usage: train.py [-h] [--save_dir SAVE_DIR] [--checkpoint CHECKPOINT]
                [--arch ARCH] [--learning_rate LEARNING_RATE]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS]
                [--drop_p DROP_P] [--gpu]
                data_dir

    positional arguments:
      data_dir              data directory of training images

    optional arguments:
      -h, --help            show this help message and exit
      --save_dir SAVE_DIR   directory to save model checkpoints; Default:
                            model_checkpoints/
      --checkpoint CHECKPOINT
                            name of checkpoint file to save; Default:
                            checkpoint.pth
      --arch ARCH           chosen model; Default: vgg16
      --learning_rate LEARNING_RATE
                            learning rate; Default: 0.001
      --hidden_units HIDDEN_UNITS
                            append a hidden layer unit - call multiple times to
                            add more layers; Default: [128, 64, 32]
      --epochs EPOCHS       number of epochs; Default - 2
      --drop_p DROP_P       dropout probability; Default - 0.2
      --gpu                 use GPU instead of CPU; Default - False

    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store", type=str,
                        help='data directory of training images')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints/',
                        help='directory to save model checkpoints; Default: model_checkpoints/')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='name of checkpoint file to save; Default: checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='chosen model; Default: vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate; Default: 0.001')
    parser.add_argument('--hidden_units', action="append", type=int, default=[],
                        help='append a hidden layer unit - call multiple times to add more layers; Default: [128, 64, 32]')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs; Default - 2')
    parser.add_argument('--drop_p', type=float, default=0.2,
                        help='dropout probability; Default - 0.2')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='use GPU instead of CPU; Default - False')
    return parser.parse_args()


if __name__ == '__main__':
    main()
