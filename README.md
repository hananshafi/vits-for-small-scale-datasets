# How to Train Vision Transformer on Small-scale Datasets?

[Hanan Gani](https://scholar.google.co.in/citations?user=XFugeQ4AAAAJ&hl=en), [Muzammal Naseer](https://muzammal-naseer.netlify.app/), and [Mohammad Yaqub](https://scholar.google.co.uk/citations?hl=en&user=9dfn5GkAAAAJ&view_op=list_works&sortby=pubdate)

#


> **Abstract:** *Vision Transformer (ViT), a radically different architecture than convolutional neural networks offers multiple advantages including design simplicity, robustness and state-of-the-art performance on many vision tasks. However, in contrast to convolutional neural networks, Vision Transformer lacks inherent inductive biases. Therefore, successful training of such models is mainly attributed to pre-training on large-scale datasets such as ImageNet with 1.2M or JFT with 300M images. This hinders the direct adaption of Vision Transformer for small-scale datasets. In this work, we show that self-supervised inductive biases can be learned directly from small-scale datasets and serve as an effective weight initialization scheme for fine tuning. This allows to train these models without large scale pre-training, changes to model architecture or loss functions. We present thorough experiments to successfully train monolithic and non-monolithic Vision Transformers on five small datasets including CIFAR10/100, CINIC-10, SVHN, and Tiny-ImageNet. Our approach consistently improves the performance while retaining their properties such as attention to salient regions and higher robustness.*
>



#
<hr>

## Contents

1. [Highlights](#Highlights)
2. [Requirements](#Requirements)
3. [Self-Supervised Pretraining](#Run-self-supervised-pretraining-with-ViT-architecture)
4. [Supervised Training](#Finetune-the-self-supervised-pretrained-checkpoint-on-the-given-dataset)
5. [Effect of Input Resolution](#Effect-of-Input-Resolution)
6. [Results](#Results)
7. [Citation](#Citation)
8. [Contact](#Contact)
9. [References](#References)


<hr>

## Highlights
1. We propose an effective two-stage framework to train ViTs on small-scale low resolution datasets from scratch. In the first stage, we introduce self-supervised weight learning scheme based on feature prediction of our low-resolution global and local views via self-distillation. In the
second stage, we fine-tune the same ViT network on the same target dataset using simply cross-entropy loss. This serves as an effective weights initialization to successfully train ViTs from scratch, thus eliminating the need for large-scale pre-training. 


<!-- <img src="assets/final_main_figure.png" height="500" width="700"> -->
<!-- ![main_figure](assets/final_main_figure.png) -->


2. Our proposed self-supervised inductive biases improve the performance of ViTs on small datasets without modifying the network architecture or loss functions.

<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Run self-supervised pretraining 

#### For Tiny-Imagenet:
With ViT architecture

```shell
python -m torch.distributed.launch --nproc_per_node=2 train_ssl.py --arch vit \
                                   --dataset Tiny_Imagenet --image_size 64 \
                                   --datapath "/path/to/tiny-imagenet/train/folder" \
                                   --patch_size 8  \
                                   --mlp_head_in 192 \
                                   --local_crops_number 8 \
                                   --local_crops_scale 0.2 0.4 \
                                   --global_crops_scale 0.5 1. 
                                   --out_dim 1024 \
                                   --batch_size_per_gpu 256  \
                                   --output_dir "/path/for/saving/checkpoints"
```
With Swin architecture

```shell
python -m torch.distributed.launch --nproc_per_node=2 train_ssl.py --arch swin \
                                   --dataset Tiny_Imagenet --image_size 64 \
                                   --datapath "/path/to/tiny-imagenet/train/folder" \
                                   --patch_size 4  \
                                   --mlp_head_in 384 \
                                   --local_crops_number 8 \
                                   --local_crops_scale 0.2 0.4 \
                                   --global_crops_scale 0.5 1. 
                                   --out_dim 1024 \
                                   --batch_size_per_gpu 256  \
                                   --output_dir "/path/for/saving/checkpoints"
```

#


#### For CIFAR based datasets:

With ViT architecture
```shell
python -m torch.distributed.launch --nproc_per_node=2 train_ssl.py --arch vit \
                                   --dataset CIFAR10 --image_size 32 \
                                   --patch_size 4  \
                                   --mlp_head_in 192  \
                                   --local_crops_number 8 \
                                   --local_crops_scale 0.2 0.5 \
                                   --global_crops_scale 0.7 1. 
                                   --out_dim 1024 \
                                   --batch_size_per_gpu 256  \
                                   --output_dir "/path/for/saving/checkpoints"
```

With Swin architecture

```shell
python -m torch.distributed.launch --nproc_per_node=2 train_ssl.py --arch swin \
                                   --dataset Tiny_Imagenet --image_size 32 \
                                   --datapath "/path/to/tiny-imagenet/train/folder" \
                                   --patch_size 2  \
                                   --mlp_head_in 384  \
                                   --local_crops_number 8 \
                                   --local_crops_scale 0.2 0.5 \
                                   --global_crops_scale 0.7 1. 
                                   --out_dim 1024 \
                                   --batch_size_per_gpu 256  \
                                   --output_dir "/path/for/saving/checkpoints"
```

``` --dataset ``` can be ``` Tiny_Imagenet/CIFAR10/CIFAR100/CINIC/SVHN ```.

``` --arch ``` can be ``` vit/swin/cait ```.

``` --local_crops_scale ``` and ``` --global_crops_scale ``` vary based on the dataset used.

``` --mlp_head_in ``` is dimension of the Vision transformer output going into Projection MLP head and varies based on the model used.


<hr>

## Finetune the self-supervised pretrained checkpoint on the given dataset
```shell
python finetune.py --arch vit  \
                   --dataset Tiny-Imagenet \
                   --datapath "/path/to/data/folder" \
                   --batch_size 256 \
                   --epochs 100 \
                   --pretrained_weights "/path/to/saved/checkpoint"
``` 
``` --arch ``` can be ```vit/swin/cait ```.
``` --datasets ``` can be ```Tiny-Imagenet/CIFAR10/CIFAR100/CINIC/SVHN ```.
Load the corresponding weights for finetuning.

<hr>

## Results
We test our approach on 5 small low resolution datasets: Tiny-Imagenet, CIFAR10, CIFAR100, CINIC10 and SVHN. We compare the results of our approach with 4 baselines: ConvNets, Scratch ViT training, [Efficient Training of Visual Transformers with Small Datasets (NIPS'21)](https://openreview.net/forum?id=SCN8UaetXx), [Vision Transformer for Small-Size Datasets (arXiv'21)](https://arxiv.org/abs/2112.13492)
#### 1. Quantitative results :
![main_results](assets/results_quantitative.PNG)

#

#### 2. Results on high resolution inputs as compared to baseline - [Efficient Training of Visual Transformers with Small Datasets (NIPS'21)](https://openreview.net/forum?id=SCN8UaetXx)
![results_nips](assets/results_nips.PNG)

#

####  3. Qualitative results - Attention to salient regions
Our proposed approach is able to capture
the shape of the salient objects more efficiently with minimal or no attention to the background as compared to the baseline approaches where the attention is more spread out in the
background and they completely fail to capture the shape of the salient object in the image.

<p align="center">
<img src="assets/atten_maps_paper.png" height="250" width="275">
</p>

<hr>

## Citation

<hr>

## Contact
Should you have any questions, please create an issue in this repository or contact at hanan.ghani@mbzuai.ac.ae
<hr>

## References
Our code is build on the repositories of [DINO](https://github.com/facebookresearch/dino) and [Vision Transformer for Small-Size Datasets](https://github.com/aanna0701/SPT_LSA_ViT). We thank them for releasing their code.

<hr>

  
