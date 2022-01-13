import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

class new_Module(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.class_idx = 0
        
        self.__name__ = "new_Module"

def set_class(net, num):
    if hasattr(net, 'class_idx'):
        net.class_idx = num
    for module in net.children():
        set_class(module, num)

def get_loss(net):
    val = 0
    if hasattr(net, 'det_bn'):
        val += net.det_bn
        #net.det_bn = 0
    for module in net.children():
        val += get_loss(module)
    return val

criterion = nn.CrossEntropyLoss()

class intra_BN_2d(new_Module):
    def __init__(self, channels, class_num=10):
        super().__init__()
        
        self.class_num = class_num
        
        self.weight = nn.Parameter(torch.ones(1,channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channels,1,1))
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels, class_num)
        
        self.bns = nn.ModuleList()
        for i in range(class_num):
            self.bns.append(nn.BatchNorm2d(channels, affine=False))
        
    def forward(self, x):
        if self.training:
            x = self.weight * self.bns[self.class_idx](x) + self.bias
            self.det_bn = criterion(self.fc(self.pool(x).squeeze()), self.class_idx*torch.ones(x.shape[0]).long().to(x.device))
        else:
            classfied = self.fc(self.pool(x).squeeze()).argmax(0).item()
            x = self.weight * self.bns[classfied](x) + self.bias
            
        
        
        '''
        else:
            norm = []
            norm_var = []
            for bn in self.bns:
                temp = bn(x)
                temp_var = temp.var().item()
                norm.append(temp)
                norm_var.append(temp_var)
            norm_var = np.array(norm_var)
            min_idx = norm_var.argmin()
            
            x = self.weight * norm[min_idx] + self.bias
        '''    
        '''
        else:
            norm = 0
            for bn in self.bns:
                norm += bn(x)
            x = self.weight * norm/self.class_num + self.bias
        '''
            
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.bn1 = intra_BN_2d(inplanes)
        self.relu = nn.GELU()
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = intra_BN_2d(planes)
        self.conv2 = conv3x3(planes, planes)
        
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
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)    

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
    
    b.train()
    set_class(b,1)
    c = b(a)
    print(c.shape)
    print(c)
    
    b.eval()
    a = torch.randn((1,3,32,32)).to(device)
    c = b(a)
    print(c.shape)
    print(c)
    
    parameter = list(b.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)