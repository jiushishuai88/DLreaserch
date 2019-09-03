#here are some cnn experiments
##argumentation:4-pixel padding random cropping,horizontal flipping
##usage:
e.g:
CUDA_VISIBLE_DEVICES=0 python -u train.py --work-path ./configs/cifar10/lenet

CUDA_VISIBLE_DEVICES=0 python -u train.py --work-path ./configs/cifar100/GePreresneXt29_32x8d --resume

