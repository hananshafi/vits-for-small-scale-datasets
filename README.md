# How to Train Vision Transformer on Small-scale Datasets?

[Hanan Gani](https://scholar.google.co.in/citations?user=XFugeQ4AAAAJ&hl=en), [Muzammal Naseer](https://muzammal-naseer.netlify.app/), and [Mohammad Yaqub](https://scholar.google.co.uk/citations?hl=en&user=9dfn5GkAAAAJ&view_op=list_works&sortby=pubdate)

#


> **Abstract:** *Vision Transformer (ViT), a radically different architecture than convolutional neural networks offers multiple advantages including design simplicity, robustness and state-of-the-art performance on many vision tasks. However, in contrast to convolutional neural networks, Vision Transformer lacks inherent inductive biases. Therefore, successful training of such models is mainly attributed to pre-training on large-scale datasets such as ImageNet with 1.2M or JFT with 300M images. This hinders the direct adaption of Vision Transformer for small-scale datasets. In this work, we show that self-supervised inductive biases can be learned directly from small-scale datasets and serve as an effective weight initialization scheme for fine tuning. This allows to train these models without large scale pre-training, changes to model architecture or loss functions. We present thorough experiments to successfully train monolithic and non-monolithic Vision Transformers on five small datasets including CIFAR10/100, CINIC-10, SVHN, and Tiny-ImageNet. Our approach consistently improves the performance while retaining their properties such as attention to salient regions and higher robustness. Our codes along with pre-trained models will be made available publicly.*

#

## Overview of Training Framework

![main_figure](assets/final_main_figure.png)

#
<hr>
<hr>

## Requirements
```shell
pip install -r requirements.txt
```
#
<hr>
<hr>

## Run self-supervised pretraining on Tiny-Imagenet with ViT architecture
```shell
python -m torch.distributed.launch --nproc_per_node=2 train_ssl.py --arch vit --image_size 64 --patch_size 8  --embed_dim 192 --num_layers 9 --num_heads 12 --local_crops_number 8 --local_crops_scale 0.2 0.4 --global_crops_scale 0.5 1. --out_dim 1024 --batch_size_per_gpu 256 --epochs 800 --output_dir /path/for/saving/checkpoints/
```


