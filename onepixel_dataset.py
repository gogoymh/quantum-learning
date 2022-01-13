import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms, datasets

class onepixel_cifar10(Dataset):
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
        
        self.range = [i for i in range(4,29)]
        self.num = 1
        
    def __getitem__(self, index):
        
        x, y = self.dataset.__getitem__(index)
        
        x = self.transform(x)
        
        var = np.random.uniform(3e-2,7e-2,1)[0]
        w1, w2 = sorted(np.random.choice(32,2,replace=False))
        h1, h2 = sorted(np.random.choice(32,2,replace=False))
        x[:,w1:w2,h1:h2] += torch.from_numpy(np.random.normal(0,var,(3,(w2-w1),(h2-h1))))
        x += torch.from_numpy(np.random.normal(0,var,(3,32,32)))
        '''
        for _ in range(self.num):
            w, h = np.random.choice(self.range,2,replace=True)
            x[0,w,h] = np.random.uniform(-1,1,1)[0]
            x[1,w,h] = np.random.uniform(-1,1,1)[0]
            x[2,w,h] = np.random.uniform(-1,1,1)[0]
        '''
        return x.clamp(-1,1), y
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    a = onepixel_cifar10(transforms.Compose([
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