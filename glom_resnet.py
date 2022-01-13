import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class GLOM_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channel, out_channel,1,1,0,bias=False)
        
    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)

        return out


class Repeat(nn.Module):
    def __init__(self, planes, repeat):
        super().__init__()
        
        self.block = GLOM_Conv2d(planes, planes)
        self.repeat = repeat
        
    def forward(self, x):
        
        for _ in range(self.repeat):
            x = self.block(x) + x
            
        return x

    

class ResNet56(nn.Module):
    def __init__(self):
        super().__init__()
        print("GLOM resnet")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = nn.Sequential(
            Repeat(64,9),
            nn.BatchNorm2d(64)
            )
        self.down1 = GLOM_Conv2d(64,64,stride=2)
        
        self.layer2 = nn.Sequential(
            Repeat(64,9),
            nn.BatchNorm2d(64)
            )
        self.down2 = GLOM_Conv2d(64,64,stride=2)
        
        self.layer3 = nn.Sequential(
            Repeat(64,9),
            nn.BatchNorm2d(64)
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):

        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.down1(x)
        
        x = self.layer2(x)
        x = self.down2(x)
        
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((2,3,32,32)).to(device)
    b = ResNet56().to(device)
    c = b(a)
    print(c.shape)
    print(c)
    
    parameter = list(b.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
