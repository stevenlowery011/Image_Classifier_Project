"""
random_image.py docstring
"""

import argparse
import json
import numpy as np
import os


def main():
    """
    docstring
    """
    in_arg = get_input_args()

    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    category = str(np.random.randint(1, len(cat_to_name)))
    directory = in_arg.path + "/" + category
    files = os.listdir(directory)
    idx = np.random.randint(0, len(files))
    image_path = directory + "/" + files[idx]
    print(image_path)
    print(cat_to_name[str(category)])


def get_input_args():
    """
    docstring
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='flowers/valid',
                        help='the path to the folder containing image categories; Default - flowers/valid')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='category to name mapping *.json file; Default - cat_to_name.json')

    return parser.parse_args()


if __name__ == '__main__':
    main()
