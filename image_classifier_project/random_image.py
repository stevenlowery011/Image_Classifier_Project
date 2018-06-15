"""
random_image.py
This script gets a random image from the directory and
returns the path to that image and the category of the image.
"""

import argparse
import json
import numpy as np
import os


def main():
    """
    Main function for random_image.py - parses the command line inputs
    and gets the image directory and the category names *.json file. 
    Pulls a random image from the directory and returns the path to that
    image along with the category name of that image. 
    Prints the path and category to the output window.
    """
    in_arg = get_input_args()

    # Get the category names from the mapping file
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Get a random category number from the classes
    category = str(np.random.randint(1, len(cat_to_name)))

    # Navigate to the directory of the category selected above
    directory = in_arg.path + "/" + category

    # List all image file names from that category folder
    files = os.listdir(directory)

    # Select a random image from the directory
    idx = np.random.randint(0, len(files))
    image_path = directory + "/" + files[idx]

    # Print the image path and its category to the output window
    print(image_path)
    print(cat_to_name[str(category)])


def get_input_args():
    """
    Parse command line arguments.

    usage: random_image.py [-h] [--path PATH] [--category_names CATEGORY_NAMES]

    optional arguments:
      -h, --help            show this help message and exit
      --path PATH           the path to the folder containing image categories;
                            Default - flowers/valid
      --category_names CATEGORY_NAMES
                            category to name mapping *.json file; Default -
                            cat_to_name.json

    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='flowers/valid',
                        help='the path to the folder containing image categories; Default - flowers/valid')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='category to name mapping *.json file; Default - cat_to_name.json')

    return parser.parse_args()


if __name__ == '__main__':
    main()
