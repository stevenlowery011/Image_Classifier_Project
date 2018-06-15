"""
predict.py
This script predicts the category of an image. First parses the
command line arguments, then loads the image from the directory
and its associated category based on the category mapping *.json file. 
Loads a checkpoint of a trained neural network model and passes the image
through the trained network and predicts the probabilities of the output 
classes. Then displays the image along with the top k most likely probabilities.
"""

import argparse
from PIL import Image
import matplotlib.pyplot as plt
import json
from image_classifier_project import *

def main():
    """
    Main function for predict.py - Loads a model checkpoint from a saved model.
    Loads the category class names from the *.json file. 
    Predicts the output along with top k probabilities.
    If the *.json file was provided, the output is predicted in category names,
    otherwise output is predicted in category index.
    Plots the image and the top k probabilities in a horizontal bar chart.
    """
    in_arg = get_input_args()

    # Load the model checkpoint
    print("\nLoading model checkpoint: {}\n".format(in_arg.checkpoint))
    model, accuracy = load_checkpoint(in_arg.checkpoint)

    # Get the category names mapping if the file was provided
    if in_arg.category_names!='':
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
        model.class_to_idx = cat_to_name

    # Predict the probabilities and top k classes
    print("Predicting image category...")
    probs_tensor, classes_tensor = predict(in_arg.input, model, in_arg.gpu, in_arg.top_k)
    
    # Convert the probabilities and classes tensors into lists
    probs = probs_tensor.tolist()[0]
    # If the category mapping file was provided, make a list of names
    if in_arg.category_names!='':
        classes = [model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (classes_tensor).tolist()[0]]
    # Otherwise create a list of index numbers
    else:
        classes = classes_tensor.tolist()[0]
    
    # Open the image
    image = process_image(Image.open(in_arg.input))

    # Plot the image and the top k probabilities in a horizontal bar chart
    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), nrows=2)
    ax1 = imshow(image[0], ax1)
    ax1.axis('off')
    if in_arg.input_category!='':
        ax1.set_title("Input Image: {}".format(in_arg.input_category))
    else:
        ax1.set_title('Input Image:')
    ax2.barh(-np.arange(in_arg.top_k), probs, tick_label=classes)
    ax2.set_title("Predicted Class:\n(model accuracy: {:.3f})".format(accuracy))
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Category')
    plt.show()


def get_input_args():
    """
    Parse command line arguments.

    usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu] [--input_category INPUT_CATEGORY]
                  input checkpoint

    positional arguments:
      input                 full path name to input image file; example:
                            flowers/valid/63/image_05876.jpg; get a random image
                            and category using random_image.py
      checkpoint            the full path to a checkpoint file *.pth of a trained
                            network; example: model_checkpoints/checkpoint.pth

    optional arguments:
      -h, --help            show this help message and exit
      --top_k TOP_K         top k results from classifier; integer number
      --category_names CATEGORY_NAMES
                            the full path to a *.json file mapping categories to
                            names; example: cat_to_name.json
      --gpu                 pass this argument to use GPU, else use CPU; default:
                            False
      --input_category INPUT_CATEGORY
                            the category of the input image (with escape character
                            preceding spaces); string; example: black-eyed\ susan

    Parameters:
        None
    Returns:
        parse_args() - data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action="store", type=str, 
                        help='full path name to input image file; example: flowers/valid/63/image_05876.jpg; get a random image and category using random_image.py')
    parser.add_argument('checkpoint', action="store", type=str, default='checkpoint.pth',
                        help='the full path to a checkpoint file *.pth of a trained network; example: model_checkpoints/checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k results from classifier; integer number')
    parser.add_argument('--category_names', type=str, default='',
                        help='the full path to a *.json file mapping categories to names; example: cat_to_name.json')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='pass this argument to use GPU, else use CPU; default: False')
    parser.add_argument('--input_category', type=str, default='',
                        help='the category of the input image (with escape character preceding spaces); string; example: black-eyed\ susan')
    return parser.parse_args()


if __name__ == '__main__':
    main()
