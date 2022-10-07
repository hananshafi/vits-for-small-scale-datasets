from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.build_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['vit', 'swin' , 'cait']


def init_parser():
    parser = argparse.ArgumentParser(description='Vit small datasets quick training script')

    # Data args
    parser.add_argument('--datapath', default='./data', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Tiny-Imagenet', 'SVHN','CINIC'], type=str, help='small dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--arch', type=str, default='vit', choices=MODELS)

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    
    parser.add_argument('--resume', default=False, help='Version')
       
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--cm',action='store_false' , help='Use Cutmix')
    
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    
    parser.add_argument('--mu',action='store_false' , help='Use Mixup')
    
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    
    parser.add_argument('--is_LSA', action='store_true', help='Locality Self-Attention')
    
    parser.add_argument('--is_SPT', action='store_true', help='Shifted Patch Tokenization')

    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")

    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--patch_size', default=4, type=int, help='patch size for ViT')

    parser.add_argument('--vit_mlp_ratio', default=2, type=int, help='MLP layers in the transformer encoder')

    return parser


def main(args):
    global best_acc1    
    
    torch.cuda.set_device(args.gpu)

    data_info = datainfo(logger, args)
    
    model = create_model(data_info['img_size'], data_info['n_classes'], args)
   
    model.cuda(args.gpu)  
        
    print(Fore.GREEN+'*'*80)
    logger.debug(f"Creating model: {model_name}")    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*'*80+Style.RESET_ALL)
    
    if os.path.isfile(args.pretrained_weights):
        model_dict = model.state_dict()
        print("loading pretrained weights . . .")
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict={k:v if v.size()==model_dict[k].size()  else  model_dict[k] for k,v in zip(model_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict, strict=False)
        #print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    if args.ls:
        print(Fore.YELLOW + '*'*80)
        logger.debug('label smoothing used')
        print('*'*80+Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()
    
    else:
        criterion = nn.CrossEntropyLoss()    
        
    if args.sd > 0.:
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*'*80+Style.RESET_ALL)         

    criterion = criterion.cuda(args.gpu)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]


    if args.cm:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Cutmix used')
        print('*'*80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Mixup used')
        print('*'*80 + Style.RESET_ALL)
    if args.ra > 1:        
        
        print(Fore.YELLOW+'*'*80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*'*80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []
    
    augmentations += [                
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(data_info['img_size'], padding=4)
            ]
    
    if args.aa == True:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Autoaugmentation used')      
        
        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [   
                CIFAR10Policy()
            ]
            
        elif 'SVHN' in args.dataset:
            print("SVHN Policy")    
            from utils.autoaug import SVHNPolicy
            augmentations += [
                SVHNPolicy()
            ]
                    
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [                
                ImageNetPolicy()
            ]
            
        print('*'*80 + Style.RESET_ALL)
    
    augmentations += [                
            transforms.ToTensor(),
            *normalize]  

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*'*80+Style.RESET_ALL)    
        
        augmentations += [     
            RandomErasing(probability = args.re, sh = args.re_sh, r1 = args.re_r1, mean=data_info['stat'][0])
            ]
    
    
    augmentations = transforms.Compose(augmentations)
      
    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    #summary(model, (3, data_info['img_size'], data_info['img_size']))
    
    print()
    print("Beginning training")
    print()
    
    lr = optimizer.param_groups[0]["lr"]
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)
    
    
    for epoch in tqdm(range(args.epochs)):
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        acc1 = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
            }, 
            os.path.join(save_path, 'checkpoint.pth'))
        
        logger_dict.print()
        
        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1
            
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(save_path, 'best.pth'))         
        
        print(f'Best acc1 {best_acc1:.2f}')
        print('*'*80)
        print(Style.RESET_ALL)        
        
        writer.add_scalar("Learning Rate", lr, epoch)
        
        
    print(Fore.RED+'*'*80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*'*80+Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler,  args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
        
    
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                output = model(images)
                loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                
                   
            else:
                output = model(images)
                
                loss = criterion(output, target)
                               
                
        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)
                
                loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                
                
            
            else:
                output = model(images)
                
                loss =  criterion(output, target)
                 
                
        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)
                
                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)
                    
                    loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                    
                    
                # Mixup
                else:
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)
                    
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam) 
                    
            else:
                output = model(images)
                
                loss = criterion(output, target) 
          
        # No Mix
        else:
            output = model(images)
                                
            loss = criterion(output, target)
            
        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(i, len(train_loader),f'[Epoch {epoch+1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f}'+' '*10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)
    
    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            
            output = model(images)
            loss = criterion(output, target)
            
            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}')
    print()        

    print(Fore.BLUE)
    print('*'*80)
    
    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)

    
    return avg_acc1


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model

    if not args.is_SPT:
        model_name += "-Base"
    else:
        print("spt present")
        model_name += "-SPT"
 
    if args.is_LSA:
        print("lsa present")
        model_name += "-LSA"
        
    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}"
    save_path = os.path.join(os.getcwd(), 'save_finetuned', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))
    
    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    
    main(args)