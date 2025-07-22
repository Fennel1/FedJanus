"""
CNN模型实现
CNN model implementation for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单的CNN模型
    适用于CIFAR-10/100等32x32彩色图像
    """
    
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块3
        x = F.relu(self.bn3(self.conv3(x)))
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
        return 256


class DeepCNN(nn.Module):
    """
    深层CNN模型
    更复杂的架构，适用于更复杂的数据集
    """
    
    def __init__(self, num_classes=10, input_channels=3):
        super(DeepCNN, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # 第一个卷积块
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 第二个卷积块
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第三个卷积块
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 第四个卷积块
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 全连接层
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.bn1(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.bn2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.bn3(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.bn4(self.conv4_2(x)))
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
        return 512


class CNNMNIST(nn.Module):
    """
    适用于MNIST/Fashion-MNIST的CNN模型
    针对28x28灰度图像优化
    """
    
    def __init__(self, num_classes=10, input_channels=1):
        super(CNNMNIST, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # 卷积块3
        x = F.relu(self.bn3(self.conv3(x)))
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