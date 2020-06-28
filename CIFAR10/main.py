import argparse
import os
import time
import shutil
import distutils
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import bnutils

from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--additive', default=True, type=lambda x:bool(distutils.util.strtobool(x)), help='use additive powers of two')
parser.add_argument('--train-alpha', default=True, type=lambda x:bool(distutils.util.strtobool(x)), help='make alpha trainable')
parser.add_argument('-wn', '--weightnorm', default=True, type=lambda x:bool(distutils.util.strtobool(x)), help='normalize weights')
parser.add_argument('-s', '--shift', default=False, type=lambda x:bool(distutils.util.strtobool(x)), help='use PS method')

parser.add_argument('--freeze-weights', dest='freeze_weights', action='store_true', help='freeze weights of conv and linear layers')
parser.add_argument('--freeze-biases', dest='freeze_biases', action='store_true', help='freeze biases of convolution and fully-connected layers')
parser.add_argument('--freeze-gamma', dest='freeze_gamma', action='store_true', help='freeze gamma of batchnorm layers')
parser.add_argument('--freeze-beta', dest='freeze_beta', action='store_true', help='freeze beta of batchnorm layers')
parser.add_argument('--no-backward-pass', dest='no_backward_pass', action='store_true', help='train using forward pass only to update running mean and var')
parser.add_argument('--update-mean-var', dest='update_mean_var', action='store_true', help='set absolute values of mean and var for batchnorm layers')

parser.add_argument('--result-dir', type=str, default='result', help='directory to log the checkpoints and weight logs to')
parser.add_argument('--print-weights', default=True, type=lambda x:bool(distutils.util.strtobool(x)), help='For printing the weights of Model (default: True)')

best_prec = 0
args = parser.parse_args()

def main():

    global args, best_prec
    use_gpu = torch.cuda.is_available()
    print(args.device)
    print('=> Building model...')
    model=None
    if use_gpu:
        float = True if args.bit == 32 else False
        if args.arch == 'res20':
            model = resnet20_cifar(float=float, additive=args.additive, train_alpha=args.train_alpha, weightnorm=args.weightnorm, shift=args.shift)
        elif args.arch == 'res56':
            model = resnet56_cifar(float=float, additive=args.additive, train_alpha=args.train_alpha, weightnorm=args.weightnorm, shift=args.shift)
        else:
            print('Architecture not support!')
            return
        if not float:
            for m in model.modules():
                if isinstance(m, QuantConv2d):
                    m.weight_quant = weight_quantize_fn(w_bit=args.bit, train_alpha=args.train_alpha, weightnorm=args.weightnorm)
                    m.act_grid = build_power_value(args.bit)
                    m.act_alq = act_quantization(args.bit, m.act_grid, train_alpha=args.train_alpha)
                if isinstance(m, ShiftConv2d):
                    m.weight_quant = weight_shift_fn(w_bit=args.bit, train_alpha=args.train_alpha, weightnorm=args.weightnorm)
                    m.act_grid = build_power_value(args.bit)
                    m.act_alq = act_quantization(args.bit, m.act_grid, train_alpha=args.train_alpha)

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        model_params = []
        for name, params in model.module.named_parameters():
            if 'act_alpha' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'wgt_alpha' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
            else:
                model_params += [{'params': [params]}]
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    fdir = args.result_dir+'/'+str(args.arch)+'_'+str(args.bit)+'bit'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init)
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found !')
            exit()
            
    if args.freeze_weights:
        model = bnutils.freeze_weights(model)
    if args.freeze_biases:
        model = bnutils.freeze_biases(model)
    if args.freeze_gamma:
        model = bnutils.freeze_gamma(model)
    if args.freeze_beta:
        model = bnutils.freeze_beta(model)

    print('=> loading cifar10 data...')
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    if args.update_mean_var:
        model = bnutils.update_mean_var(model, trainloader)

    if args.evaluate:
        validate(testloader, model, criterion)
        model.module.show_params()
        return
    writer = SummaryWriter(comment=fdir.replace(args.result_dir, ''))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # model.module.record_weight(writer, epoch)
        if epoch%10 == 1:
            model.module.show_params()
        # model.module.record_clip(writer, epoch)
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)
        writer.add_scalar('test_acc', prec, epoch)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        print('best acc: {:1f}'.format(best_prec))
        
        if (args.print_weights):
            os.makedirs(os.path.join(fdir, 'weights_logs'), exist_ok=True)
            with open(os.path.join(fdir, 'weights_logs', 'weights_log_' + str(epoch) + '.txt'), 'w') as weights_log_file:
                with redirect_stdout(weights_log_file):
                    # Log model's state_dict
                    print("Model's state_dict:")
                    # TODO: Use checkpoint above
                    for param_tensor in model.state_dict():
                        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                        print(model.state_dict()[param_tensor])
                        print("")        
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.no_backward_pass is False:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % 2 == 0:
        #     model.module.show_params()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))
        
    if (state['epoch']-1)%10 == 0:
        os.makedirs(os.path.join(fdir, 'checkpoints'), exist_ok=True)
        shutil.copyfile(filepath, os.path.join(fdir, 'checkpoints', 'checkpoint_' + str(state['epoch']-1) + '.pth.tar'))  


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
