from torch.utils.data import Dataset
import os
import random
from torchvision.io import read_image
import torch
from torchvision import transforms
import math

class MaskedFaceDataset(Dataset):

    def __init__(self, 
                 path : str,
                 height : int = 244, 
                 width : int = 244):        
        def filter_func(folder):
            return len(os.listdir(path + folder + '/')) >= 2

        self.id_list = list(filter(filter_func, os.listdir(path)))
        self.path = path
        self.mean = torch.Tensor([0.5360, 0.4703, 0.4324])  # Dataset mean
        self.std = torch.Tensor([0.2720, 0.2469, 0.2537])  # Dataset std
        self.height = height
        self.width = width


    def __len__(self):
        pass


    def __getitem__(self, 
                    index : int):
        pass


    def transformation(self, 
                       img : torch.Tensor) -> torch.Tensor:
        
        normalize = transforms.Normalize(self.mean, self.std)
        resize = transforms.Resize((self.height, self.width))
        transform = transforms.Compose([normalize, resize])

        return transform(img)


# Data set used for training the network
class MaskedFaceDatasetTraining(MaskedFaceDataset):

    def __init__(self, 
                 path : str, 
                 height : int = 244, 
                 width : int = 244):
        super().__init__(path, height, width)


    def __len__(self) -> int:
        return len(self.id_list)


    def __getitem__(self, 
                    index : int) -> tuple[torch.Tensor, torch.Tensor]:

        id = self.id_list[index] #str with the name
        dir_path = self.path + id + '/'
        img_list = os.listdir(dir_path)
        img_sampled = random.sample(img_list, 2)
        img_1 = read_image(dir_path + img_sampled[0]).float()/255
        img_2 = read_image(dir_path + img_sampled[1]).float()/255


        return self.transformation(img_1), self.transformation(img_2)


# Dataset used at inference time using the k-nn classifier
class MaskedFaceDatasetInference(MaskedFaceDataset):

    def __init__(self, 
                 path : str, 
                 height : int = 244, 
                 width : int = 244, 
                 id_list : list = None):
        super().__init__(path, height, width)
        self.img_list = []
        self.label = []
        if id_list:
          '''
          If None id_list is computed from the id in the folder.
          Otherwise it can be passed so we are sure to have the same 
          id list in the train and test set of the k-NN.
          '''
          self.id_list = id_list
        for id in self.id_list:
            dir_path = self.path + id + '/'
            imgs = os.listdir(dir_path)
            for img_name in imgs:
              self.img_list.append(dir_path + img_name)
              self.label.append(id)


    def __len__(self) -> int:
        return len(self.id_list)


    def __getitem__(self, 
                    index : int) -> tuple[torch.Tensor, str]:

        id = self.label[index] #str with the name
        img = read_image(self.img_list[index]).float()/255

        return self.transformation(img), id



class MaskedFaceDatasetNewSampler(Dataset):
    '''
    This version of dataset tries to exploit all the images in each epoch.
    In the first version for each epochs, we sampled only a couple of images for
    each id, now we sample n couple for each id where n is the number of couple
    that can be obtained sampling from the folder without replacing.
    '''

    def __init__(self, 
                 path : str,
                 height : int = 244, 
                 width : int = 244):        
                 
        self.id_list = []
        self.id_listdir = os.listdir(path)
        for id in self.id_listdir:
            id_path = path + id + '/'
            n_img = len(os.listdir(id_path))
            for _ in range(math.floor(n_img/2)):
                self.id_list.append(id)

        self.path = path
        self.mean = torch.Tensor([0.5360, 0.4703, 0.4324])  # Dataset mean
        self.std = torch.Tensor([0.2720, 0.2469, 0.2537])  # Dataset std
        self.height = height
        self.width = width
        
        self.empty_dict()


    def __len__(self) -> int:
        return len(self.id_list)


    def __getitem__(self, 
                    index : int) -> tuple[torch.Tensor, torch.Tensor]:

        id = self.id_list[index] #str with the name
        dir_path = self.path + id + '/'
        img_list = os.listdir(dir_path)
        img_list = list(set(img_list) - set(self.old_sample[id]))
        img_sampled = random.sample(img_list, 2)
        self.old_sample[id].append(img_sampled[0])
        self.old_sample[id].append(img_sampled[1])

        img_1 = read_image(dir_path + img_sampled[0]).float()/255
        img_2 = read_image(dir_path + img_sampled[1]).float()/255

        return self.transformation(img_1), self.transformation(img_2)
    

    def transformation(self, 
                       img : torch.Tensor) -> torch.Tensor:
        
        normalize = transforms.Normalize(self.mean, self.std)
        resize = transforms.Resize((self.height, self.width))
        transform = transforms.Compose([normalize, resize])

        return transform(img)
    

    def empty_dict(self) -> None:
        self.old_sample = {id : [] for id in self.id_listdir}

