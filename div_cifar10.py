import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms, datasets
import pickle

class Numberset(Dataset):
    def __init__(self, num):
        super().__init__()
        
        self.len = num
        
    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return self.len

class USDataset(Dataset):
    def __init__(self, class_idx, path="class_cifar10_%d.pkl", transform=None):
        super().__init__()
        
        self.class_idx = class_idx
        self.path = path
        self.transform = transform
        
        if os.path.isfile(path % class_idx):
            with open(path % class_idx, 'rb') as f:
                self.data = pickle.load(f)    
        else:
            temp_set = datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=None,
                        )
        
        
    def __getitem__(self, index):
        
        input_img = imread(os.path.join(self.path1, "%07d.jpg" % (index + self.idx_init)), as_gray=True)
        target_img = imread(os.path.join(self.path2, "%07d.jpg" % (index + self.idx_init)), as_gray=True)
        
        input_img = np.expand_dims(input_img, axis=2).astype('float32')
        target_img = np.expand_dims(target_img, axis=2).astype('float32')
        
        seed = np.random.randint(2147483647)
        
        if self.input_transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            input_img = self.input_transform(input_img)

            input_img = self.normalize(input_img)
            
        if self.target_transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            target_img = self.target_transform(target_img)
        
        return input_img, target_img
    
    def __len__(self):
        return self.len