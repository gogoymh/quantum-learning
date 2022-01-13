import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class LG_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        #print("local global")
        
        if stride == 2:
            self.conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
                )
            self.conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
                )
            self.conv3 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
                )
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
            self.conv2 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        
        local_ = self.sigmoid(self.conv1(x))
        B,C,H,W = local_.shape
        global_ = self.softmax(self.conv2(x).reshape(B,C,-1)).reshape(B,C,H,W)
        
        gate = local_ + global_        
        value = self.conv3(x)
        
        out = gate * value

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
        self.conv1 = LG_Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.conv2 = LG_Conv2d(planes, planes, bias=False)
        
        
        if inplanes == 16:
            self.bn1 = nn.LayerNorm((16,32,32))
        elif inplanes == 32:
            self.bn1 = nn.LayerNorm((32,16,16))
        else:
            self.bn1 = nn.LayerNorm((64,8,8))
        
        if planes == 16:
            self.bn2 = nn.LayerNorm((16,32,32))
        elif planes == 32:
            self.bn2 = nn.LayerNorm((32,16,16))
        else:
            self.bn2 = nn.LayerNorm((64,8,8))
        '''
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        '''
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        return out


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

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
 
        if stride != 1 or self.inplanes != planes * block.expansion:
            if planes * block.expansion == 32:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0),
                    #nn.BatchNorm2d(planes * block.expansion)
                    nn.LayerNorm((32,16,16))
                    )
            elif planes * block.expansion == 64:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0),
                    #nn.BatchNorm2d(planes * block.expansion)
                    nn.LayerNorm((64,8,8))
                    )

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
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

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
    
