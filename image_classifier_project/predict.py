"""
predict.py docstring
"""

import argparse
from PIL import Image
import matplotlib.pyplot as plt
import json
from image_classifier_project import *

def main():
    """
    docstring
    """
    in_arg = get_input_args()

    print("\nLoading model checkpoint: {}\n".format(in_arg.checkpoint))
    model, accuracy = load_checkpoint(in_arg.checkpoint)

    if in_arg.category_names!='':
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
        model.class_to_idx = cat_to_name

    print("Predicting image category...")
    probs_tensor, classes_tensor = predict(in_arg.input, model, in_arg.gpu, in_arg.top_k)
    
    probs = probs_tensor.tolist()[0]
    if in_arg.category_names!='':
        classes = [model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (classes_tensor).tolist()[0]]
    else:
        classes = classes_tensor.tolist()[0]
    
    image = process_image(Image.open(in_arg.input))
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
    docstring
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action="store", type=str, 
                        help='path to input image file')
    parser.add_argument('checkpoint', action="store", type=str, default='checkpoint.pth',
                        help='a checkpoint file *.pth of a trained network')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k results from classifier')
    parser.add_argument('--category_names', type=str, default='',
                        help='a *.json file mapping categories to names')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='pass this argument to use GPU, else use CPU')
    parser.add_argument('--input_category', type=str, default='',
                        help='the category of the input image')
    return parser.parse_args()


if __name__ == '__main__':
    main()
