"""
模型管理器实现
Model Manager implementation for federated learning
"""

import torch
import torch.nn as nn
import copy
import pickle
from typing import Dict, Any, Optional, Union
from collections import OrderedDict

from .lenet import LeNet, LeNetCIFAR
from .cnn import SimpleCNN, DeepCNN, CNNMNIST
from .resnet import (
    ResNet18, ResNet34, ResNet50,
    ResNet18CIFAR, ResNet34CIFAR,
    ResNet18MNIST
)


class ModelManager:
    """
    模型管理器类
    负责动态创建模型、参数管理和序列化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型管理器
        
        Args:
            config: 模型配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型注册表
        self.model_registry = {
            # LeNet系列
            'lenet': LeNet,
            'lenet_cifar': LeNetCIFAR,
            
            # CNN系列
            'cnn': SimpleCNN,
            'simple_cnn': SimpleCNN,
            'deep_cnn': DeepCNN,
            'cnn_mnist': CNNMNIST,
            
            # ResNet系列
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'resnet50': ResNet50,
            'resnet18_cifar': ResNet18CIFAR,
            'resnet34_cifar': ResNet34CIFAR,
            'resnet18_mnist': ResNet18MNIST,
        }
    
    def create_model(self, model_name: str, input_shape: tuple, num_classes: int) -> nn.Module:
        """
        动态创建模型
        
        Args:
            model_name: 模型名称
            input_shape: 输入形状 (channels, height, width)
            num_classes: 分类数量
            
        Returns:
            创建的模型实例
            
        Raises:
            ValueError: 当模型名称不支持时
        """
        model_name_lower = model_name.lower()
        
        if model_name_lower not in self.model_registry:
            raise ValueError(f"不支持的模型类型: {model_name}. 支持的模型: {list(self.model_registry.keys())}")
        
        model_class = self.model_registry[model_name_lower]
        input_channels = input_shape[0] if len(input_shape) == 3 else 1
        
        # 根据数据集和模型类型选择合适的模型变体
        if model_name_lower == 'lenet':
            if input_shape[1:] == (32, 32):  # CIFAR数据集
                model = LeNetCIFAR(num_classes=num_classes, input_channels=input_channels)
            else:  # MNIST数据集
                model = LeNet(num_classes=num_classes, input_channels=input_channels)
        elif model_name_lower in ['cnn', 'simple_cnn']:
            if input_shape[1:] == (28, 28):  # MNIST数据集
                model = CNNMNIST(num_classes=num_classes, input_channels=input_channels)
            else:  # CIFAR数据集
                model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
        elif model_name_lower == 'resnet18':
            if input_shape[1:] == (32, 32):  # CIFAR数据集
                model = ResNet18CIFAR(num_classes=num_classes, input_channels=input_channels)
            elif input_shape[1:] == (28, 28):  # MNIST数据集
                model = ResNet18MNIST(num_classes=num_classes, input_channels=input_channels)
            else:  # 标准ImageNet尺寸
                model = ResNet18(num_classes=num_classes, input_channels=input_channels)
        else:
            # 直接使用指定的模型类
            model = model_class(num_classes=num_classes, input_channels=input_channels)
        
        model.to(self.device)
        return model
    
    def get_model_parameters(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取模型参数
        
        Args:
            model: PyTorch模型
            
        Returns:
            模型参数字典
        """
        return {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def get_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取完整的模型状态字典（包括BatchNorm的running statistics）
        
        Args:
            model: PyTorch模型
            
        Returns:
            模型状态字典
        """
        return {name: param.clone().detach() for name, param in model.state_dict().items()}
    
    def set_model_parameters(self, model: nn.Module, parameters: Dict[str, torch.Tensor]) -> None:
        """
        设置模型参数
        
        Args:
            model: PyTorch模型
            parameters: 参数字典
        """
        model_dict = model.state_dict()
        
        # 验证参数键是否匹配
        param_keys = set(parameters.keys())
        model_keys = set(model_dict.keys())
        
        if param_keys != model_keys:
            missing_keys = model_keys - param_keys
            unexpected_keys = param_keys - model_keys
            
            if missing_keys:
                print(f"警告: 缺少参数键: {missing_keys}")
            if unexpected_keys:
                print(f"警告: 意外的参数键: {unexpected_keys}")
        
        # 更新模型参数
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name].copy_(param.to(self.device))
    
    def set_model_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        设置完整的模型状态字典（包括BatchNorm的running statistics）
        
        Args:
            model: PyTorch模型
            state_dict: 状态字典
        """
        # 将状态字典中的参数移动到正确的设备
        device_state_dict = {name: param.to(self.device) for name, param in state_dict.items()}
        model.load_state_dict(device_state_dict)
    
    def get_model_gradients(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取模型梯度
        
        Args:
            model: PyTorch模型
            
        Returns:
            模型梯度字典
        """
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
            else:
                gradients[name] = torch.zeros_like(param)
        return gradients
    
    def set_model_gradients(self, model: nn.Module, gradients: Dict[str, torch.Tensor]) -> None:
        """
        设置模型梯度
        
        Args:
            model: PyTorch模型
            gradients: 梯度字典
        """
        for name, param in model.named_parameters():
            if name in gradients:
                if param.grad is None:
                    param.grad = gradients[name].clone().to(self.device)
                else:
                    param.grad.copy_(gradients[name].to(self.device))
    
    def serialize_parameters(self, parameters: Dict[str, torch.Tensor]) -> bytes:
        """
        序列化模型参数
        
        Args:
            parameters: 参数字典
            
        Returns:
            序列化后的字节数据
        """
        # 将tensor转换为CPU并序列化
        cpu_parameters = {name: param.cpu() for name, param in parameters.items()}
        return pickle.dumps(cpu_parameters)
    
    def deserialize_parameters(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        反序列化模型参数
        
        Args:
            data: 序列化的字节数据
            
        Returns:
            参数字典
        """
        parameters = pickle.loads(data)
        # 将参数移动到正确的设备
        return {name: param.to(self.device) for name, param in parameters.items()}
    
    def save_model(self, model: nn.Module, filepath: str) -> None:
        """
        保存模型到文件
        
        Args:
            model: PyTorch模型
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self.config,
            'model_class': model.__class__.__name__
        }, filepath)
    
    def load_model(self, filepath: str, model_name: str, input_shape: tuple, num_classes: int) -> nn.Module:
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            model_name: 模型名称
            input_shape: 输入形状
            num_classes: 分类数量
            
        Returns:
            加载的模型
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 创建模型实例
        model = self.create_model(model_name, input_shape, num_classes)
        
        # 加载参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def clone_model(self, model: nn.Module) -> nn.Module:
        """
        克隆模型
        
        Args:
            model: 源模型
            
        Returns:
            克隆的模型
        """
        cloned_model = copy.deepcopy(model)
        cloned_model.to(self.device)
        return cloned_model
    
    def get_model_size(self, model: nn.Module) -> int:
        """
        获取模型参数数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            参数总数
        """
        return sum(p.numel() for p in model.parameters())
    
    def get_trainable_parameters(self, model: nn.Module) -> int:
        """
        获取可训练参数数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def freeze_layers(self, model: nn.Module, layer_names: list) -> None:
        """
        冻结指定层的参数
        
        Args:
            model: PyTorch模型
            layer_names: 要冻结的层名称列表
        """
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def unfreeze_layers(self, model: nn.Module, layer_names: list) -> None:
        """
        解冻指定层的参数
        
        Args:
            model: PyTorch模型
            layer_names: 要解冻的层名称列表
        """
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    def get_supported_models(self) -> list:
        """
        获取支持的模型列表
        
        Returns:
            支持的模型名称列表
        """
        return list(self.model_registry.keys())
    
    def model_summary(self, model: nn.Module, input_shape: tuple) -> str:
        """
        生成模型摘要信息
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
            
        Returns:
            模型摘要字符串
        """
        total_params = self.get_model_size(model)
        trainable_params = self.get_trainable_parameters(model)
        
        summary = f"""
模型摘要:
========
模型类型: {model.__class__.__name__}
输入形状: {input_shape}
总参数数: {total_params:,}
可训练参数: {trainable_params:,}
设备: {self.device}
========
        """
        
        return summary.strip()