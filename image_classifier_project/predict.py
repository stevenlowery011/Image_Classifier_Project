"""
predict.py docstring
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
    parser.add_argument('input', action="store", type=str, 
                        help='path to input image file')
    parser.add_argument('checkpoint', action="store", type=str, default='checkpoint.pth',
                        help='a checkpoint file *.pth of a trained network')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k results from classifier')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='a *.json file mapping categories to names')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='pass this argument to use GPU, else use CPU')
    return parser.parse_args()



if __name__ == '__main__':
    main()
