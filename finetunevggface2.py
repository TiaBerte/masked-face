import torch
import torchvision
from torchvision import transforms
from barlowTwins import BarlowTwins
from utils import adjust_learning_rate, load_state_dict, get_mean_and_std, LARS

from models.resnet50 import resnet50
from utils import load_state_dict

from pathlib import Path
import argparse
import json
import sys
import time
from torch import nn
import torchvision.transforms as transforms



"""Train = 2397
val = 300
Test = 300
"""

N_IDENTITY = 8631
model = resnet50(num_classes=N_IDENTITY, include_top=True)#False)
load_state_dict(model, PATH+'weights/resnet50_ft_weight.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device)



train_path = PATH + 'dataset/train'
val_path = PATH + 'dataset/val'
test_path = PATH + 'dataset/test'


dataset = torchvision.datasets.ImageFolder(train_path, transform=transforms.ToTensor())

dataset_mean = torch.Tensor([0.5360, 0.4703, 0.4324])
dataset_std = torch.Tensor([0.2720, 0.2469, 0.2537])

normalize = transforms.Normalize(dataset_mean, dataset_std)
resize = transforms.Resize((244, 244))

transform = transforms.Compose([transforms.ToTensor(), normalize, resize])

data_train = torchvision.datasets.ImageFolder(train_path, transform=transform)

data_val = torchvision.datasets.ImageFolder(val_path, transform=transform)

data_test = torchvision.datasets.ImageFolder(test_path, transform=transform)

num_workers = 2
size_batch_train = 64
size_batch_val = 2 * size_batch_train

loader_train = torch.utils.data.DataLoader(data_train, batch_size=size_batch_train, 
                                           shuffle=True, 
                                           pin_memory=True, 
                                           num_workers=num_workers)

loader_val = torch.utils.data.DataLoader(data_val, batch_size=size_batch_val, 
                                         shuffle=False,
                                         num_workers=num_workers)


parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')



def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)

    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    #dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    data_train = torchvision.datasets.ImageFolder(train_path, transform=resize)
    '''
    Change the sampler
    '''
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')




