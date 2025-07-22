"""
LeNet模型实现
LeNet model implementation for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    LeNet-5 模型实现
    适用于MNIST、Fashion-MNIST等28x28灰度图像
    """
    
    def __init__(self, num_classes=10, input_channels=1):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 84


class LeNetCIFAR(nn.Module):
    """
    适用于CIFAR-10/100的LeNet变体
    针对32x32彩色图像优化
    """
    
    def __init__(self, num_classes=10, input_channels=3):
        super(LeNetCIFAR, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 卷积块2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 卷积块3
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 128