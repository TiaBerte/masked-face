from torch.utils.data import Dataset
import os
import random
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms



class maskedFaceDataset(Dataset):

    def __init__(self, path, height=244, width=244):

        self.path = path
        self.id_list = os.listdir(self.path)
        self.mean = torch.Tensor([0.5360, 0.4703, 0.4324]) # Dataset mean and std
        self.std = torch.Tensor([0.2720, 0.2469, 0.2537])
        self.height = height
        self.width = width


    def __len__(self):
        return len(self.id_list)


    def __getitem__(self, index):

        id = self.id_list[index] #str with the name
        dir_path = self.path + id + '/'
        img_list = os.listdir(dir_path)
        img_sampled = random.sample(img_list, 2)
        img_1 = read_image(dir_path + img_sampled[0]).float()/255
        img_2 = read_image(dir_path + img_sampled[1]).float()/255


        return self.transformation(img_1), self.transformation(img_2)


    def transformation(self, img):
        
        normalize = transforms.Normalize(self.mean, self.std)
        resize = transforms.Resize((self.height, self.width))
        transform = transforms.Compose([normalize, resize])

        return transform(img)


