#!/bin/sh
# 
# 

echo 'Training vgg11 network. Please wait about 10-15 minutes...'
stdbuf -oL python train.py flowers --checkpoint checkpoint_vgg11.pth --arch vgg11 --epochs 4 --gpu | tee train_output.txt
