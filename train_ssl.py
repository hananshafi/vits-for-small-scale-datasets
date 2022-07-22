# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Self-supervised training script for low resolution small-scale datasets"""


import argparse
from concurrent.futures import process
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from utils import utils_ssl as utils
from projection_head import MLPHead
from functools import partial
from models.vit import VisionTransformer
from models.swin import SwinTransformer
from models.cait import cait_models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('SSL for low resolution dataset', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small','vit', 'swin', 'cait'] \
                + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of
        the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=64, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default=9, type=int, help=""" No. of layers of ViT """)
    parser.add_argument('--num_heads',default=12, type=int, help=""" No. of heads in attention layer
                                                                                 in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--drop_rate',default=0., type=float, help=""" dropout """)
    parser.add_argument('--vit_init_value',default=0.1, type=float, help=""" initialisation values of ViT """)
    parser.add_argument('--use_ape',default=False, type=bool, help=""" Absolute position embeddings """)
    parser.add_argument('--use_rpb',default=False, type=bool, help=""" Relative position embeddings """)
    parser.add_argument('--use_shared_rpb',default=False, type=bool, help=""" Shared Relative position embeddings """)
    parser.add_argument('--use_mean_pooling',default=False, type=bool, help=""" Shared Relative position embeddings """)

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=30, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.5, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=10, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.2, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--dataset', default='Tiny-Imagenet', type=str,
        choices=['Tiny-Imagenet', 'CIFAR10', 'CIFAR100','CINIC','SVHN'],
        help='Please specify path to the training data.')
    parser.add_argument('--data_path', default='./data', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentation(
        args
    )
    if args.dataset == 'Tiny_Imagenet':
        dataset =  datasets.ImageFolder(
           root=os.path.join("/path/to/tiny-imagenet/train"), transform=transform)
    elif args.dataset = "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=True,
                                       download=True, transform=transform)
    elif args.dataset = "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root=args.datapath, train=True,
                                       download=True, transform=transform)
    elif args.dataset = "CINIC":
        dataset = datasets.ImageFolder("/path/to/cinic/train",transform=transform)
    
    elif args.dataset = "SVHN":
        dataset = torchvision.datasets.SVHN(root=args.datapath,split='train',
                                         download=True, transform=transform)
    

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    
    # Custom models
    if args.arch == 'vit':

        student = VisionTransformer(img_size=[args.image_size],
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            num_classes=0,
            embed_dim=args.embed_dim,
            depth=args.num_layers,
            num_heads=args.num_heads,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=args.qkv_bias,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        teacher = VisionTransformer(img_size=[args.image_size],
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            num_classes=0,
            embed_dim=args.embed_dim,
            depth=args.num_layers,
            num_heads=args.num_heads,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=args.qkv_bias,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))



    elif args.arch =='swin':

        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if args.image_size == 32 else 4

        student = SwinTransformer(img_size=args.image_size,num_classes=0
        window_size=window_size, patch_size=patch_size, embed_dim=args.embed_dim, depths=[2, 6, 4], num_heads=[3, 6, 12],
        mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.drop_path_rate)

        teacher = SwinTransformer(img_size=args.image_size,num_classes=0,
        window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],
        mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.drop_path_rate)


    elif args.arch == 'custom_cait':
        patch_size =  4 if args.image_size == 32 else 8
        student = cait_models(
        img_size= args.image_size,patch_size=patch_size, embed_dim=192, depth=24, num_heads=4, mlp_ratio=2, qkv_bias=True,num_classes=0,
                drop_path_rate=args.drop_path_rate,norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5,depth_token_only=2)
        teacher = cait_models(
        img_size= args.image_size,patch_size=patch_size, embed_dim=192, depth=24, num_heads=4, mlp_ratio=2, qkv_bias=True,num_classes=0
                drop_path_rate=args.drop_path_rate,norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5,depth_token_only=2)

    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions

    student = utils.MultiCropWrapper(student, MLPHead(args.embed_dim, args.out_dim, args.use_bn_in_head))
                                            
    teacher = utils.MultiCropWrapper(
        teacher, MLPHead(
        args.embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        
    )
    )


    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
   
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)


    # teacher and student start with the same weights
    filtered_student_state_dict = {k: v for k, v in student.module.state_dict().items() if k in teacher_without_ddp.state_dict().keys()}
    teacher_without_ddp.load_state_dict(filtered_student_state_dict)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    view_pred_loss = ViewPredLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 512. ,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        view_pred_loss=view_pred_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting SSL training !")
    for epoch in range(start_epoch, args.epochs):

        
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, view_pred_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'view_pred_loss': view_pred_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, view_pred_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images,_) in enumerate(metric_logger.log_every(data_loader, 10, header)):   #add mask here
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # teacher and student forward passes + compute loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output  = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output  = student(images)

            total_loss = view_pred_loss(student_output, teacher_output, epoch)
            loss = total_loss.pop('loss')
            loss_view = total_loss.pop('ce_loss')

        if not math.isfinite(loss.item()): 
            print("Loss is {}, View Pred loss is {}, stopping training".format(loss.item(),loss_view.item()
                                                                                           ), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter

            names_q, params_q, names_k, params_k = [], [], [], []
            for name_q, param_q in student.module.named_parameters():
                names_q.append(name_q)
                params_q.append(param_q)
            for name_k, param_k in teacher_without_ddp.named_parameters():
                names_k.append(name_k)
                params_k.append(param_k)
            names_common = list(set(names_q) & set(names_k))
            params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
            params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            


            # for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
            #     param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(view_pred_loss=view_pred_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class ViewPredLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, in_channels=3, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.in_channels = in_channels
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)


        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)


        n_loss_terms = 0
        total_loss = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                    

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        total_loss = dict( ce_loss=total_loss, loss=total_loss)
        self.update_center(teacher_output)
        return total_loss



    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentation(object):
    def __init__(self, args):
        if args.dataset=="CIFAR10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif args.dataset == "CIFAR100":
            mean, std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        elif args.dataset == "SVHN":
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        elif args.dataset == "CINIC":
            mean, std = (0.47889522, 0.47227842, 0.43047404),(0.24205776, 0.23828046, 0.25874835)
        elif args.dataset == "Tiny-Imagenet":
            mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)


        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
	    transforms.Normalize(mean, std),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=args.global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=args.global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = args.local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size//2, scale=args.local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSL for low resolution dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)