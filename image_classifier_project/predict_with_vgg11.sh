#!/bin/sh
# 
# 

classifier='model_checkpoints/checkpoint_vgg11.pth'
stdbuf -oL python random_image.py > random_image_output.txt
image_path="$(sed '1!d' random_image_output.txt)"
image_class="$(sed '2!d' random_image_output.txt)"
image_class=${image_class//' '/'_'}
stdbuf -oL python predict.py $image_path $classifier --category_names cat_to_name.json --gpu --input_category $image_class > predict_output.txt
