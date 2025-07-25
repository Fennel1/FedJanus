"""
ResNet模型实现
ResNet model implementation for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet基础块
    用于ResNet-18和ResNet-34
    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    ResNet瓶颈块
    用于ResNet-50, ResNet-101, ResNet-152
    """
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet模型基类
    """
    
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 512 * self.layer4[-1].expansion


def ResNet18(num_classes=10, input_channels=3):
    """ResNet-18模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


def ResNet34(num_classes=10, input_channels=3):
    """ResNet-34模型"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels)


def ResNet50(num_classes=10, input_channels=3):
    """ResNet-50模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels)


class ResNetCIFAR(nn.Module):
    """
    适用于CIFAR数据集的ResNet变体
    针对32x32图像优化
    """
    
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 64 * self.layer3[-1].expansion


def ResNet18CIFAR(num_classes=10, input_channels=3):
    """适用于CIFAR的ResNet-18"""
    return ResNetCIFAR(BasicBlock, [2, 2, 2], num_classes, input_channels)


def ResNet34CIFAR(num_classes=10, input_channels=3):
    """适用于CIFAR的ResNet-34"""
    return ResNetCIFAR(BasicBlock, [3, 4, 6], num_classes, input_channels)


class ResNetMNIST(nn.Module):
    """
    适用于MNIST/Fashion-MNIST的ResNet变体
    针对28x28灰度图像优化
    """
    
    def __init__(self, block, num_blocks, num_classes=10, input_channels=1):
        super(ResNetMNIST, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 64 * self.layer3[-1].expansion


def ResNet18MNIST(num_classes=10, input_channels=1):
    """适用于MNIST的ResNet-18"""
    return ResNetMNIST(BasicBlock, [2, 2, 2], num_classes, input_channels)