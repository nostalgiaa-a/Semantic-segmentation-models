import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
import os

def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class train_dataset(data.Dataset):   
    def __init__(self, data_path='', size_w=256, size_h=256, flip=1):
        super(train_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/imgs/') if is_image_file(x)]
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        
    def __getitem__(self, index):
        initial_path = os.path.join(self.data_path + '/imgs/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/masks/', self.list[index])
        assert os.path.exists(semantic_path)
        try:
            initial_image = Image.open(initial_path).convert('RGB')
            # semantic_image = Image.open(semantic_path).point(lambda i: i * 80).convert('RGB')
            semantic_image = Image.open(semantic_path)                                           

        except OSError:
            return None, None, None

        label_n = np.array(semantic_image)      #将像素值103的目标转为255作二分类，根据数据集不同做调整
        label = np.zeros(label_n.shape) 
        label[label_n==103] = 255

        semantic_image = label

        semantic_image = Image.fromarray(semantic_image/255)

        initial_image = initial_image.resize((self.size_w, self.size_h), Image.BILINEAR)    
        semantic_image = semantic_image.resize((self.size_w, self.size_h), Image.BILINEAR)
        # initial_image,semantic_image=data_augment(initial_image,semantic_image)  



        ### normalize to [0,1] ###
        #semantic_image = semantic_image+1/2
        #semantic_image = np.float(semantic_image)
        #semantic_image = (semantic_image-semantic_image.min())/(semantic_image.max()-semantic_image.min())

        # if self.flip == 1:
        #     a = random.random()
        #     if a < 1 / 3:
        #         initial_image = initial_image.transpose(Image.FLIP_LEFT_RIGHT)
        #         semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
        #     else:
        #         if a < 2 / 3:
        #             initial_image = initial_image.transpose(Image.ROTATE_90)
        #             semantic_image = semantic_image.transpose(Image.ROTATE_90)
        
        to_tensor = transforms.ToTensor()
        initial_image = to_tensor(initial_image)
        semantic_image = to_tensor(semantic_image)
     
        
        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)
