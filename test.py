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
from dataload import MaskedFaceDatasetInference
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

path = './dataset/'
train_path = path + 'train/'
test_path = path + 'test/'

train_set = MaskedFaceDatasetInference(train_path)
test_set = MaskedFaceDatasetInference(test_path)


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
    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        model.load_state_dict(ckpt['model'])
    
    # Togliere i sampler ???
    sampler_train = torch.utils.data.distributed.DistributedSampler(train_set)
    sampler_test = torch.utils.data.distributed.DistributedSampler(test_set)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader_train = DataLoader(train_set, batch_size=per_device_batch_size, 
                                        pin_memory=True, 
                                        num_workers=args.workers,
                                        sampler=sampler_train
                            )
    loader_test = DataLoader(test_set, batch_size=per_device_batch_size, 
                                        num_workers=args.workers,
                                        sampler=sampler_test
                            )

    emb_train = []
    id_train = []
    data_bar = tqdm(loader_train, desc=f"Training embeddings generation")
    for img, id in tqdm(data_bar):
        emb = model.projector(model.backbone(img.cuda()))
        emb_train.append(emb.cpu().detach().numpy())
        id_train.append(id)

    emb_train = np.vstack(emb_train) 
    id_train = np.hstack(id_train) 

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(emb_train, id_train)

    pred_test = []
    id_test = []
    data_bar = tqdm(loader_test, desc=f"Predicting test accuracy")
    for img, id in tqdm(data_bar):
        emb_test = model.projector(model.backbone(img.cuda()))
        pred = knn.predict(emb_test.cpu().detach().numpy())
        pred_test.append(pred)
        id_test.append(id)


    pred_test = np.hstack(pred_test)
    id_test = np.hstack(id_test)
    acc = accuracy_score(pred_test, id_test)
    print("#"*30)
    print(f"Test accuracy : {acc:.2f}")
    print("#"*30)
