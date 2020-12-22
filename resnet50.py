#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.models as models

import os

model_url ='https://download.pytorch.org/models/resnet50-19c8e357.pth'


def conv1x1(num_in, num_out, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(num_in, num_out, kernel_size=1, stride=stride, bias=False)


def conv3x3(num_in, num_out, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(num_in, num_out, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Resnet50(nn.Module):
    def __init__(self, pretrained=None, max_pool=True, feature_dim=128, *args, **kwargs):
        super(Resnet50, self).__init__(*args, **kwargs)

        self.pretrained = pretrained
        resnet50 = models.resnet50()
        self.conv1 = resnet50.conv1         # 7x7, 64, stride 2
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool     # 3x3, stride 2 max pooling
        self.layer1 = create_layer(num_in=64, num_mid=64, num_block=3, stride=1)
        self.layer2 = create_layer(num_in=256, num_mid=128, num_block=4, stride=2)
        self.layer3 = create_layer(num_in=512, num_mid=256, num_block=6, stride=2)
        self.layer4 = create_layer(num_in=1024, num_mid=512, num_block=3, stride=1)

        if max_pool:
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained is not None:
            print("Load pretrained weight", end='...')
            if os.path.exists(pretrained):
                new_state = torch.load(pretrained)
            else:
                print('{} not exist'.format(pretrained))
                exit()
                # new_state = model_zoo.load_url(model_url)
                # new_state = load_state_dict_from_url(model_url, progress=True)
            state_dict = self.state_dict()
            for k, v in new_state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.load_state_dict(state_dict)
            print('Done!')

        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        self.conv1_out = self.relu(x)
        x = self.maxpool(self.conv1_out)

        self.conv2_out = self.layer1(x)
        self.conv3_out = self.layer2(self.conv2_out)
        self.conv4_out = self.layer3(self.conv3_out)
        self.conv5_out = self.layer4(self.conv4_out)

        h = self.pooling(self.conv5_out)
        h = torch.flatten(h, start_dim=1)
        z = self.g(h)

        return F.normalize(h, dim=-1), F.normalize(z, dim=-1)


class Bottleneck(nn.Module):
    def __init__(self, num_in, num_mid, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        num_out = num_mid * 4
        self.conv1 = conv1x1(num_in=num_in, num_out=num_mid)
        self.bn1 = nn.BatchNorm2d(num_mid)
        self.conv2 = conv3x3(num_in=num_mid, num_out=num_mid, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_mid)
        self.conv3 = conv1x1(num_in=num_mid, num_out=num_out)
        self.bn3 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if stride != 1 or num_in != num_out:
            self.downsample = nn.Sequential(
                conv1x1(num_in=num_in, num_out=num_out, stride=stride),
                nn.BatchNorm2d(num_out),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def create_layer(num_in, num_mid, num_block, stride=1):
    num_out = num_mid * 4
    layers = []
    layers.append(Bottleneck(num_in=num_in, num_mid=num_mid, stride=stride))
    for _ in range(1, num_block):
        layers.append(Bottleneck(num_in=num_out, num_mid=num_mid, stride=1))
    return nn.Sequential(*layers)


# Model Test
if __name__ == "__main__":
    os.system('pwd')
    net = Resnet50(pretrained=True, num_class=751)

    input = torch.randn(32, 3, 256, 128)
    out, logit = net(input)
    print(net.conv1_out.size())
    print(net.conv2_out.size())
    print(net.conv3_out.size())
    print(net.conv4_out.size())
    print(net.conv5_out.size())
    print(out.size())
    print(logit.size())