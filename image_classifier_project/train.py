"""
train.py docstring
"""

import argparse

def main():
    in_arg = get_input_args()
    print(in_arg)



def get_input_args():
    """
    docstring
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action="store", type=str, default='flowers/train',
                        help='data directory of training images')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints/',
                        help='directory to save model checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=list, default=[128, 64, 32],
                        help='hidden layer units')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
