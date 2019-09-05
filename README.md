#here are some cnn experiments
##argumentation:4-pixel padding random cropping,horizontal flipping
##usage:
e.g:
CUDA_VISIBLE_DEVICES=0 python -u train.py --work-path ./configs/cifar10/lenet

CUDA_VISIBLE_DEVICES=0 python -u train.py --work-path ./configs/cifar100/GePreresneXt29_32x8d --resume


| architecture          | params | batch size | epoch | C100 test acc (%) |
| :-------------------- | :----: | :--------: | :---: | :---------------: |
| Lecun                 |69656   |    128     |  250  |       33.25       |
| alexnet               |8658372 |    128     |  250  |       42.28       |
| vgg11                 |9802596 |    128     |  250  |       69.36       |
| vgg13                 |9987492 |    128     |  250  |       72.84       |
| vgg16                 |15299748|    128     |  250  |       72.13       |
| vgg19                 |20612004|    128     |  250  |       72.06       |
| restNet19             |15484644|    128     |  250  |       70.98       |
| preresneXt29_32x8d    |7690660 |    128     |  250  |       79.26       |
| dense_bc_100_12       |800032  |    128     |  250  |       76.53       |
| BamPreresneXt29_32x8d |8213350 |    128     |  250  |     **79.71**     |
| GePreresneXt29_32x8d  |9593764 |    128     |  250  |       77.85       |
| sePreresneXt29_32x8d  |7690660 |    128     |  250  |       79.71       |
| SKPreresneXt29_16x8d  |4865732 |    128     |  250  |       78.47       |
