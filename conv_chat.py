import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class ECA_Layer(nn.Module):
    def __init__(self, channels):
        super(ECA_Layer, self).__init__()
        
        kernel = math.ceil((math.log(channels,2)/2 + 0.5))
        if kernel % 2 == 0:
            kernel -= 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return y

class Conv2d_att(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        #self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_channel)
        #self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_channel)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
        self.att = ECA_Layer(in_channel)

    def forward(self, x):

        gate = self.conv1(x)
        gate = self.att(gate)
                
        value = self.conv2(x)
        
        out = gate * value
        
        return out
'''
class Conv2d_att(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):

        query = self.conv1(x).squeeze(3)
        key = self.conv2(x).squeeze(3)
        gate = self.softmax(torch.matmul(query,key.transpose(-1,-2)))
        
        value = self.conv3(x)
        B,C,H,W = value.shape
        value = value.reshape(B,C,-1)
        
        out = torch.matmul(gate,value)
        out = out.reshape(B,C,H,W)
        
        return out
'''
class Multi_head_conv_att(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        #print("multi head")
        
        self.head1 = Conv2d_att(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.head2 = Conv2d_att(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.head3 = Conv2d_att(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.head4 = Conv2d_att(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)       
        
        cat_channel = int(in_channel*4)
        #self.conv1x1 = nn.Conv2d(cat_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(cat_channel, cat_channel, kernel_size=3, stride=1, padding=1, bias=False, groups=cat_channel),
            nn.Conv2d(cat_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
    def forward(self, x):
        
        cat = torch.cat((self.head1(x), self.head2(x), self.head3(x), self.head4(x)), dim=1)
        out = self.conv1x1(cat)
        
        return out
        

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
        
        if inplanes == 16:
            self.bn0 = nn.LayerNorm((16,32,32))
        elif inplanes == 32:
            self.bn0 = nn.LayerNorm((32,16,16))
        else:
            self.bn0 = nn.LayerNorm((64,8,8))
        
        #self.bn0 = norm_layer(inplanes)
        self.conv0 = Multi_head_conv_att(inplanes, planes, stride=stride)        
                
        if planes == 16:
            self.bn1 = nn.LayerNorm((16,32,32))
            self.bn2 = nn.LayerNorm((16,32,32))
        elif planes == 32:
            self.bn1 = nn.LayerNorm((32,16,16))
            self.bn2 = nn.LayerNorm((32,16,16))
        else:
            self.bn1 = nn.LayerNorm((64,8,8))
            self.bn2 = nn.LayerNorm((64,8,8))
        
        #self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        #self.conv1 = conv3x3(planes, planes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=planes),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
        #self.bn2 = norm_layer(planes)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=planes),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            )
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.bn0(x)
        out = self.conv0(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        identity = out
        
        out = self.bn1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

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
        self.avgpool = nn.Sequential(
            nn.LayerNorm((64,8,8)),
            nn.AdaptiveAvgPool2d((1, 1))
            )

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
