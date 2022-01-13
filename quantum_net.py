import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class net1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(3072,100, bias=False),
            nn.LeakyReLU(),
            nn.Linear(100,100, bias=False),
            nn.LeakyReLU(),
            nn.Linear(100,10, bias=False)
            )
        
    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        
        return x

class net2(nn.Module):
    def __init__(self):
        super().__init__()
        
        print("net2")
        
        self.integer = nn.Parameter(torch.ones(1) * 100000)
        
        #weight = torch.randn(328200)#.softmax(-1) * self.integer
        #self.weight = nn.Parameter(weight)
        
        self.gate = nn.Parameter(torch.randn(328200, 1000))
        #self.gate = nn.Parameter(torch.randn(328500, 1000))
        self.weight = nn.Parameter(torch.randn(1000))
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.weight2 = nn.Parameter(torch.randn(328200))
        
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        
        self.relu = nn.GELU()
        
        self.dropout = nn.Dropout(p=0.0)
        
    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        
        #weight = self.weight.softmax(-1) * self.integer
        weight = torch.matmul(self.gate.softmax(-1), self.weight) + self.bias
        #weight = self.weight2
        
        weight1 = weight[0:307200].reshape(100,3072)
        weight2 = weight[307200:317200].reshape(100,100)
        weight3 = weight[317200:327200].reshape(100,100)
        weight4 = weight[327200:328200].reshape(10,100)
        
        '''
        bias1 = weight[328200:328300]
        bias2 = weight[328300:328400]
        bias3 = weight[328400:328500]
        x = self.dropout(self.relu(F.linear(x, weight1, bias=bias1)))
        x = self.dropout(self.relu(F.linear(x, weight2, bias=bias2)))
        x = self.dropout(self.relu(F.linear(x, weight3, bias=bias3)))
        '''
        x = self.dropout(self.relu(self.bn1(F.linear(x, weight1, bias=None))))
        x = self.dropout(self.relu(self.bn2(F.linear(x, weight2, bias=None))))
        x = self.dropout(self.relu(self.bn3(F.linear(x, weight3, bias=None))))
        x = F.linear(x, weight4, bias=None)
        
        return x


class net3(nn.Module):
    def __init__(self, base=50000):
        super().__init__()
        
        print("net3")
        
        num_param = 24112
        self.gate = nn.Parameter(torch.randn(num_param, base), requires_grad=False)
        self.weight = nn.Parameter(torch.randn(base))
        self.bias = nn.Parameter(torch.zeros(1))
        
        #self.weight2 = nn.Parameter(torch.randn(num_param))
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.relu = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x): #(B, 3, 32, 32)
        batch, _, _, _ = x.shape
        #weight = torch.matmul(self.gate.softmax(-1), self.weight) + self.bias
        weight = torch.matmul(self.gate, self.weight) + self.bias
        #weight = self.weight2
        
        '''
        #weight = (weight - weight.mean())/weight.std()
        
        weight1 = weight[0:432].reshape(16,3,3,3)
        weight2 = weight[432:5040].reshape(32,16,3,3)
        weight3 = weight[5040:23472].reshape(64,32,3,3)
        weight4 = weight[23472:24112].reshape(10,64)
        '''
        weight1 = (weight[0:432] - weight[0:432].mean().detach())/weight[0:432].std().detach()
        weight2 = (weight[432:5040] - weight[432:5040].mean().detach())/weight[432:5040].std().detach()
        weight3 = (weight[5040:23472] - weight[5040:23472].mean().detach())/weight[5040:23472].std().detach()
        weight4 = (weight[23472:24112] - weight[23472:24112].mean().detach())/weight[23472:24112].std().detach()
        
        weight1 = weight1.reshape(16,3,3,3)
        weight2 = weight2.reshape(32,16,3,3)
        weight3 = weight3.reshape(64,32,3,3)
        weight4 = weight4.reshape(10,64)
        
        
        weight1 = weight1.repeat(batch,1,1,1,1)
        weight1 = weight1.view(batch * 16, 3, 3, 3)
        x = x.reshape(1, batch * 3, 32, 32)
        x = F.conv2d(x, weight1, bias=None, stride=2, padding=1, groups=batch)
        x = self.relu(self.bn1(x.reshape(batch,16,16,16))) # (B, 16, 16, 16)
        
        weight2 = weight2.repeat(batch,1,1,1,1)
        weight2 = weight2.view(batch * 32, 16, 3, 3)
        x = x.reshape(1, batch * 16, 16, 16)
        x = F.conv2d(x, weight2, bias=None, stride=2, padding=1, groups=batch)
        x = self.relu(self.bn2(x.reshape(batch,32,8,8))) # (B, 32, 8, 8)
        
        weight3 = weight3.repeat(batch,1,1,1,1)
        weight3 = weight3.view(batch * 64, 32, 3, 3)
        x = x.reshape(1, batch * 32, 8, 8)
        x = F.conv2d(x, weight3, bias=None, stride=2, padding=1, groups=batch)
        x = self.relu(self.bn3(x.reshape(batch,64,4,4))) # (B, 64, 4, 4)

        x = self.pool(x)
        x = torch.flatten(x,1)
        x = F.linear(x, weight4, bias=None)
        
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((2,3,32,32)).to(device)
    b = net3().to(device)
    c = b(a)
    print(c.shape)
    print(c)










