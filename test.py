import os
import random
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import transforms

path = 'C:/Users/Mattia/Documents/GitHub/masked-face/dataset/train/'
list_name = os.listdir(path)
n = len(list_name)
#print(path + list_name[0] + '/')
img_list = os.listdir(path + list_name[0] + '/')
#print(img_list)
img_sampled = random.sample(img_list, 2)
#print(img_sampled)
img_1 = read_image(path + list_name[0] + '/' + img_sampled[0])
#print(img_1)
#plt.imshow(img_1.permute(1, 2, 0))
#plt.show()
img_1 = img_1.float()/255
normalize = transforms.Normalize(0.5, 0.1)
resize = transforms.Resize((3, 3))
transform = transforms.Compose([normalize, resize])
trasnformed = transform(img_1)
print(trasnformed)