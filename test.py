import torch
from barlowTwins import BarlowTwins
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataload import MaskedFaceDatasetInference
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import pandas as pd

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
parser.add_argument('--projector', default='8192', type=str)
                #metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                help='print frequency')
parser.add_argument('--checkpoint-path', default='./checkpoint/', type=Path,
                metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--backbone_lr', default=0, type=float, 
                help='Learning rate used for fine tuning the backbone, disabled by default')

parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

path = './dataset/'
train_path = path + 'train_knn/'
test_path = path + 'test_knn/'

train_set = MaskedFaceDatasetInference(train_path)
test_set = MaskedFaceDatasetInference(test_path, id_list=train_set.id_list)


def main_worker(args):

    gpu = torch.device('cuda:0')
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    model = BarlowTwins(args).cuda(gpu)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    weights_path = Path(args.checkpoint_path)
    if (weights_path).is_file():
        print('Loading weights ...')
        ckpt = torch.load(weights_path,
                          map_location='cpu')
        corrected_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(corrected_dict)
        del ckpt
        del corrected_dict
    
    per_device_batch_size = 16
    loader_train = DataLoader(train_set, batch_size=per_device_batch_size,
                                        num_workers=2,
                                        pin_memory=True,
                                        #sampler=sampler_train
                            )
    loader_test = DataLoader(test_set, batch_size=per_device_batch_size, 
                                        num_workers=2,
                                        #sampler=sampler_test
                            )

    emb_train = []
    id_train = []
    data_bar = tqdm(loader_train, desc=f"Training embeddings generation")
    for img, id in data_bar:
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
    for img, id in data_bar:
        emb_test = model.projector(model.backbone(img.cuda()))
        pred = knn.predict(emb_test.cpu().detach().numpy())
        pred_test.append(pred)
        id_test.append(id)


    pred_test = np.hstack(pred_test)
    id_test = np.hstack(id_test)
    acc = accuracy_score(pred_test, id_test)
    print("#"*30)
    print(f"Test accuracy : {100*acc:.4f}")
    print("#"*30)
    df = pd.DataFrame({'id' : id_test, 'pred' : pred_test})
    ethnicity = pd.read_csv('/content/masked-face/ethnicity_test.csv', index_col=False)
    ethnicity_dict = dict(zip(ethnicity.Name, ethnicity.Ethnicity))
    df['ethnicity'] = [ethnicity_dict[df['id'].iloc[i]] for i in range(len(df))]
    df['correct'] = [ 1 if df['id'].iloc[i] == df['pred'].iloc[i] else 0 for i in range(len(df))]
    grouped = df[['ethnicity', 'correct']].groupby(by='ethnicity')
    accuracy_ethnicity = grouped.sum()/grouped.count()
    print(accuracy_ethnicity)