# CNN Classifier
This repo includes easy-to-read scripts to train, validate and evaluate **your own dataset** using any of the most popular architectures, including: AlexNet, VGG, SqueezeNet, ResNet, and DenseNet, in any of their configurations (eg. DenseNet201, ResNet152, VGG16_BN, etc).

The whole code is written in:
<p align="center"><img width="40%" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" /></p>

## Requirements:
- Package requirements are the same for pytorch.
- Works either on python 2 or 3.
- The whole code will assume you have your dataset in `./data` with subfolders: `train`, `val`, and `test`. If you need to change this, just modify `data_loader.py`.

## Usage:
`./main.py --kwargs`

kwargs:
- *batch_size* [default=128]
- *num_epochs* [default=59]
- *num_epochs_decay* [default=60]
- *stop_training* [default=3]
- *num_workers* [default=4] 
- *model* [default='densenet201']
- *TEST* [default=False]

### Example:
`./main.py --batch_size=16 --model=resnet152`

### Misc
It trains using pretrained weights from Imagenet. 
