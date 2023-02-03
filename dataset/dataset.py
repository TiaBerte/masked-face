import torch
import torchvision
from torchvision import transforms






N_IDENTITY = 8631
train_path = PATH + 'dataset/train'
val_path = PATH + 'dataset/val'
test_path = PATH + 'dataset/test'


#dataset = torchvision.datasets.ImageFolder(train_path, transform=transforms.ToTensor())

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
