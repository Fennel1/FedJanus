# 模型模块
# Model modules

from .lenet import LeNet, LeNetCIFAR
from .cnn import SimpleCNN, DeepCNN, CNNMNIST
from .resnet import (
    ResNet18, ResNet34, ResNet50,
    ResNet18CIFAR, ResNet34CIFAR,
    ResNet18MNIST
)
from .model_manager import ModelManager

__all__ = [
    'LeNet', 'LeNetCIFAR',
    'SimpleCNN', 'DeepCNN', 'CNNMNIST',
    'ResNet18', 'ResNet34', 'ResNet50',
    'ResNet18CIFAR', 'ResNet34CIFAR',
    'ResNet18MNIST',
    'ModelManager'
]