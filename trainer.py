import argparse
import os
import random
import shutil
import time
import math
import warnings
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model.CNV import cnv
# to prevent PIL error while training
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PIL.Image as PILI

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
parser.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=123456, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')
parser.add_argument('--pretrained-path', default=None, type=str, help='Path to the pretrained model')
parser.add_argument('--bit-width', default=4, type=int, help='Bit width of the model (level of quantization)')
parser.add_argument('--first-layer-bit-width', default=8, type=int, help='Bit width of the first layer of the model')

best_acc1 = 0


def main():
    args = parser.parse_args()

    prec_name = "_q{}".format(args.bit_width)
    experiment_name = '{}{}_{}'.format("cnv", prec_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir_path = os.path.join(args.experiments, experiment_name)

    args.checkpoints_dir_path = os.path.join(output_dir_path, 'checkpoints')
    os.mkdir(output_dir_path)
    os.mkdir(args.checkpoints_dir_path)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.rank)
        torch.cuda.manual_seed(args.seed + args.rank)
        np.random.seed(seed=args.seed + args.rank)
        random.seed(args.seed + args.rank)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    global best_acc1
    args.gpu = "cuda:0"

    model = cnv(args.bit_width, args.bit_width, args.first_layer_bit_width)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cudnn.benchmark = True

    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_transforms_list = [transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()]
    transform_train = transforms.Compose(train_transforms_list)

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers, pin_memory=True)

    valset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        print("Training started at: ", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        print("Training stopped at: ", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.checkpoints_dir_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.clip_weights(-1,1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, checkpoints_dir):
    torch.save(state, os.path.join(checkpoints_dir, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(checkpoints_dir, 'checkpoint.pth.tar'), os.path.join(checkpoints_dir, 'model_best.pth.tar'))

def checkImage(path):
    try:
        im = PILI.open(path)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close() #reload is necessary in my case
        return True
    except:
        print("This couldn't be loaded" + path)
        return False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()