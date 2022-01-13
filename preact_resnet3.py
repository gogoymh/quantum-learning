import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class All_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.weight1 = nn.Parameter(torch.randn(out_channel,in_channel,kernel_size*kernel_size))
        self.weight2 = nn.Parameter(torch.randn(out_channel,in_channel,kernel_size*kernel_size))
        self.weight3 = nn.Parameter(torch.randn(out_channel,in_channel,kernel_size*kernel_size)*0.001)
        
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        
        #self.alpha = nn.Parameter(torch.ones(out_channel, 1, 1, 1)*0.001)
        #self.beta = nn.Parameter(torch.zeros(out_channel, 1, 1, 1))
        
        self.stride = stride
        self.padding = padding
        
        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, input, max_value, min_value):
        #gate = self.weight1.sigmoid()#
        #gate = (self.weight1 * self.weight3).softmax(dim=-1).reshape(-1)
        gate = torch.matmul(self.weight1.permute(0,2,1), self.weight2).softmax(dim=-1)
        #weight = self.weight2 * gate 
        weight = torch.matmul(gate, self.weight3.permute(0,2,1)).permute(0,2,1)
        weight = weight.reshape(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        
        #weight = (self.weight-min_value)/(max_value-min_value)
        #weight = weight * self.alpha + self.beta

        out = F.conv2d(input, weight, bias=None, stride=self.stride, padding=self.padding)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = All_Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.conv2 = All_Conv2d(planes, planes, bias=False)
        
        #self.bn1 = nn.BatchNorm2d(inplanes)
        #self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_tuple):
        x, max_value, min_value = x_tuple

        identity = x
        
        #out = self.bn1(x)
        out = self.relu(x)
        out = self.conv1(out, max_value, min_value)
        
        #out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out, max_value, min_value)

        if self.downsample is not None:
            identity = self.downsample(x, max_value, min_value)

        out += identity
        
        return (out, max_value, min_value)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = All_Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #self.fc = All_Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        
    def get_minmax(self):
        self.min_value = 0
        self.max_value = 0
        for m in self.modules():
            if isinstance(m, All_Conv2d):
                if self.min_value > m.weight.min().item():
                    self.min_value = m.weight.min().item()
                if self.max_value < m.weight.max().item():
                    self.max_value = m.weight.max().item()
            
            '''
            if isinstance(m, All_Linear):
                if self.min_value > m.weight.min().item():
                    self.min_value = m.weight.min().item()
                if self.max_value < m.weight.max().item():
                    self.max_value = m.weight.max().item()
                    
                if self.min_value > m.bias.min().item():
                    self.min_value = m.bias.min().item()
                if self.max_value < m.bias.max().item():
                    self.max_value = m.bias.max().item()
            '''

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
 
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = All_Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0)
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        self.get_minmax()
        #scale = (self.max_value - self.min_value) * 2
        
        x = self.conv1(x, self.max_value, self.min_value)
        
        x, _, _ = self.layer1((x, self.max_value, self.min_value))
        x, _, _ = self.layer2((x, self.max_value, self.min_value))
        x, _, _ = self.layer3((x, self.max_value, self.min_value))
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)#, self.max_value, self.min_value)

        return x

def _resnet(inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    return model

def resnet56(pretrained=False, progress=True, **kwargs):
    return _resnet(BasicBlock, [9,9,9], pretrained, progress, **kwargs)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((2,3,32,32)).to(device)
    b = resnet56().to(device)
    
    c = b(a)
    print(c.shape)
    print(c)
    
    parameter = list(b.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
    
