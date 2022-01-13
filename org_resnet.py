import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
'''
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
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        #self.relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.GELU()
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
'''
    
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
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        #self.relu = nn.LeakyReLU(inplace=True)
        self.relu = nn.GELU()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.conv1 = Our_Conv2d(inplanes, planes, stride=stride)
        
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        #self.conv2 = Our_Conv2d(planes, planes)
        
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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
                #norm_layer(planes * block.expansion),
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

class ECA_Layer(nn.Module):
    def __init__(self, channels, kernel=3):
        super(ECA_Layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x*y.expand_as(x)

class DSC2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()

        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        
    def forward(self, x, y): # x is input, y is cumulative concatenation
        out = self.depthwise(x)
        out = self.pointwise(out)

        cat = torch.cat((out, y), dim=1)
        return out, cat

class Our_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, inter_capacity=None, num_scale=4, kernel=3, stride=1):
        super().__init__()
        if inter_capacity is None:
            inter_capacity = out_ch//num_scale
        
        self.init = nn.Conv2d(in_ch, inter_capacity, 1, stride, 0, bias=False)
        
        self.scale_module = nn.ModuleList()
        for _ in range(num_scale-1):
            self.scale_module.append(DSC2d(inter_capacity, inter_capacity, 3, 1, 1))
        
        self.channel_attention = ECA_Layer(num_scale*inter_capacity, kernel)
        
    def forward(self, x):
        out = self.init(x)

        cat = out
        for operation in self.scale_module:
            out, cat = operation(out, cat)
        
        cat = self.channel_attention(cat)
        
        return cat


class kernel_trick(nn.Module):
    def __init__(self, stride=1, padding=1):
        super().__init__()
        
        self.kernel = torch.FloatTensor([[1,1,1,1],
                                         [0,0,0,1],
                                         [0,0,1,0],
                                         [0,1,0,0],
                                         [1,0,0,0],
                                         [0,0,1,1],
                                         [0,1,0,1],
                                         [0,1,1,0],
                                         [1,0,0,1],
                                         [1,0,1,0],
                                         [1,1,0,0],
                                         [0,1,1,1],
                                         [1,0,1,1],
                                         [1,1,0,1],
                                         [1,1,1,0],
                                         [1,1,1,1]])
        
        self.param_diff = nn.Parameter(torch.randn(16,1,4))
        self.param_same = nn.Parameter(torch.randn(16,15,1))
        
        self.stride = stride
        self.padding = padding
        
        if self.stride == 2:
            self.down = nn.Conv2d(16,16, 3,2,1, bias=False)
        
    def forward(self, x):
        kernel = self.kernel.repeat(16,1,1).to(x.device)
        param = torch.cat((self.param_diff, self.param_same.expand(-1,-1,4)), dim=1)
        #param = (param - param.mean().detach())/param.std().detach()
        param = kernel * param
        param = param.reshape(16,16,2,2)
        
        if self.stride == 2:
            x = self.down(x)
        B, _, H, W = x.shape
        param = param.repeat(B,1,1,1,1)
        param = param.view(B * 16, 16, 2, 2)
        x = x.reshape(1, B * 16, H, W)
        x = F.conv2d(x, param, bias=None, stride=1, padding=self.padding, groups=B)
        n_H = H + 2*self.padding - 1
        x = x.reshape(B,16,n_H,n_H)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((5,3,32,32)).to(device)
    b = resnet56().to(device)
    c = b(a)
    print(c.shape)
    print(c)
    
    parameter = list(b.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
    
    d = kernel_trick(1,0).to(device)
    e = torch.randn((5,16,33,33)).to(device)
    f = d(e)
    print(f.shape)