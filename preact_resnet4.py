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
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        #self.scale = nn.Parameter(torch.ones(1))
        
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        #weight = (self.weight-self.weight.mean())/self.weight.std()
        out = F.conv2d(input, self.weight, bias=None, stride=self.stride, padding=self.padding)

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
        
        #self.scale = nn.Parameter(torch.ones(1))
        self.scale1 = nn.Parameter(torch.ones(1))
        self.scale2 = nn.Parameter(torch.ones(1))
        self.scale3 = nn.Parameter(torch.ones(1))
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        #out = x.clamp(-1,1)
        out = torch.min(torch.max(x, -self.scale1.clamp(min=0)), self.scale1.clamp(min=0))
        out = self.relu(out)
        out = self.conv1(out)
        
        #out = out.clamp(-1,1)
        out = torch.min(torch.max(out, -self.scale2.clamp(min=0)), self.scale2.clamp(min=0))
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)#.clamp(-1,1)
            identity = torch.min(torch.max(identity, -self.scale3.clamp(min=0)), self.scale3.clamp(min=0))

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

        self.conv1 = All_Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        
    def sum_scale(self):
        sum_scale = 0
        for m in self.modules():
            if isinstance(m, BasicBlock):
                sum_scale += m.scale1
                sum_scale += m.scale2
                sum_scale += m.scale3
                
        return sum_scale
        
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
    
    c, d = b(a)
    print(c.shape)
    print(c)
    print(d)
    
    parameter = list(b.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
    
