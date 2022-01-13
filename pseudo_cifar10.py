import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms, datasets

class pseudo_cifar10(Dataset):
    def __init__(self, transform):
        super().__init__()
        
        self.dataset = datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=None,
                        )
        
        self.transform = transform
        self.len = self.dataset.__len__()
        
        self.pseudo_label = torch.zeros((self.len))
        
    def __getitem__(self, index):
        
        x, y_real = self.dataset.__getitem__(index)
        
        x = self.transform(x)
        
        y_pseudo = self.pseudo_label[index]

        return x, y_real, y_pseudo
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    a = pseudo_cifar10(transforms.Compose([
                                #transforms.RandomCrop(32, padding=4),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                ))
    
    b, c = a.__getitem__(7)
    
    import matplotlib.pyplot as plt
    
    b = b * 0.5 + 0.5
    b = b.numpy().transpose(1,2,0)
    plt.imshow(b)