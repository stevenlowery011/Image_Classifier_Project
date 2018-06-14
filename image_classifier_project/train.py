"""
train.py docstring
"""

import argparse
from image_classifier_project import *


def main():
    """
    docstring
    """
    in_arg = get_input_args()
    if in_arg.hidden_units==[]:
        in_arg.hidden_units=[128, 64, 32];
    save_path = in_arg.save_dir + in_arg.checkpoint
    hyperparameters = {'architecture': in_arg.arch,
                       'hidden_layers': in_arg.hidden_units,
                       'dropout_probability': in_arg.drop_p,
                       'learnrate': in_arg.learning_rate,
                       'epochs': in_arg.epochs}
    model = get_model(hyperparameters['architecture'])
    model, train_dataloader, test_dataloader, total_steps = model_config(in_arg.data_dir, model, hyperparameters['hidden_layers'], hyperparameters['dropout_probability'])
    print("\n")
    print("Transfer model:               {}".format(hyperparameters['architecture']))
    print("Hidden layers:                {}".format(hyperparameters['hidden_layers']))
    print("Learning rate:                {}".format(hyperparameters['learnrate']))
    print("Dropout probability:          {}".format(hyperparameters['dropout_probability']))
    print("Epochs:                       {}".format(hyperparameters['epochs']))
    print("Training data:                {}".format(in_arg.data_dir + '/train'))
    print("Validation data:              {}".format(in_arg.data_dir + '/test'))
    print("Checkpoint will be saved to:  {}".format(save_path))
    model_accuracy = network_train(model, hyperparameters['epochs'], hyperparameters['learnrate'], train_dataloader, test_dataloader, total_steps, in_arg.gpu)
    save_checkpoint(save_path, model, hyperparameters['architecture'], model.classifier.hidden_layers[0].in_features, model.classifier.output.out_features, hyperparameters, model_accuracy)



def get_input_args():
    """
    docstring
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
