import torch
from barlowTwins import BarlowTwins
from utils import adjust_learning_rate, LARS
from pathlib import Path
import argparse
import json
import sys
import os
import signal
import subprocess
import time
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataload import MaskedFaceDatasetTraining, MaskedFaceDatasetNewSampler


parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-4096-8192', type=str)
                #metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--backbone_lr', default=0, type=float, 
                help='Learning rate used for fine tuning the backbone, disabled by default')
parser.add_argument('--dataset', default='standard', type=str, choices=['standard', 'new_sampler']
                help='Type of dataset to use')



def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()

    def handle_sigusr1(signum, frame):
        os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
        exit()
    signal.signal(signal.SIGUSR1, handle_sigusr1)
    cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]
    args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
    args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
    args.dist_url = f'env://'
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    print(args.rank)
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

    path = './dataset/'
    train_path = path + 'train/'
    val_path = path + 'val/'

    if args.dataset == 'new_sampler':
        train_set = MaskedFaceDatasetNewSampler(train_path)
        val_set = MaskedFaceDatasetNewSampler(val_path)
    else:
        train_set = MaskedFaceDatasetTraining(train_path)
        val_set = MaskedFaceDatasetTraining(val_path)

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

    sampler_train = torch.utils.data.distributed.DistributedSampler(train_set)
    sampler_val = torch.utils.data.distributed.DistributedSampler(val_set)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader_train = DataLoader(train_set, batch_size=per_device_batch_size, 
                                        pin_memory=True, 
                                        num_workers=args.workers,
                                        sampler=sampler_train
                            )
    loader_val = DataLoader(val_set, batch_size=per_device_batch_size, 
                                        num_workers=args.workers,
                                        sampler=sampler_val
                            )                


    start_time = time.time()
    min_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        sampler_val.set_epoch(epoch)

        if isinstance(train_set, MaskedFaceDatasetNewSampler):
            train_set.empty_dict()
        
        data_bar = tqdm(loader_train, desc=f"Train Epoch {epoch}")
        for step, (y1, y2)  in enumerate(data_bar, start=epoch * len(loader_train)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader_train, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # compute val loss
        val_loss = torch.zeros(1, device=gpu)
        val_bar = tqdm(loader_val)
        for step, (val1, val2) in enumerate(val_bar):
            val1 = val1.cuda(gpu, non_blocking=True)
            val2 = val2.cuda(gpu, non_blocking=True)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    val_loss += model.forward(val1, val2)
        val_loss /= len(loader_val)
        
        torch.distributed.all_reduce(val_loss)
        val_loss /= args.world_size

        # print stats
        if args.rank == 0:
            stats = dict(epoch=epoch, step=step,
                            lr_weights=optimizer.param_groups[0]['lr'],
                            lr_biases=optimizer.param_groups[1]['lr'],
                            loss=loss.item(),
                            val_loss=val_loss.item(),
                            time=int(time.time() - start_time))
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

            # save best
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(state, args.checkpoint_dir / 'best_checkpoint.pth')


if __name__ == '__main__':
    main()
