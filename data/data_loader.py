"""
数据加载器模块
支持CIFAR-10、CIFAR-100、MNIST、Fashion-MNIST数据集的加载和缓存
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Any


class DatasetLoader:
    """数据集加载器类，支持多种标准数据集"""
    
    def __init__(self, data_dir: str = "./datasets"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据集存储目录
        """
        self.data_dir = data_dir
        self.supported_datasets = ["CIFAR-10", "CIFAR-100", "MNIST", "Fashion-MNIST"]
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 定义数据集配置
        self.dataset_configs = {
            "CIFAR-10": {
                "dataset_class": torchvision.datasets.CIFAR10,
                "num_classes": 10,
                "input_shape": (3, 32, 32),
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            },
            "CIFAR-100": {
                "dataset_class": torchvision.datasets.CIFAR100,
                "num_classes": 100,
                "input_shape": (3, 32, 32),
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            },
            "MNIST": {
                "dataset_class": torchvision.datasets.MNIST,
                "num_classes": 10,
                "input_shape": (1, 28, 28),
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            },
            "Fashion-MNIST": {
                "dataset_class": torchvision.datasets.FashionMNIST,
                "num_classes": 10,
                "input_shape": (1, 28, 28),
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
            }
        }
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            tuple: (train_data, train_labels, test_data, test_labels)
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}. 支持的数据集: {self.supported_datasets}")
        
        print(f"正在加载数据集: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        dataset_class = config["dataset_class"]
        transform = config["transform"]
        
        try:
            # 加载训练集
            train_dataset = dataset_class(
                root=self.data_dir,
                train=True,
                download=True,
                transform=transform
            )
            
            # 加载测试集
            test_dataset = dataset_class(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
            
            print(f"数据集 {dataset_name} 加载成功")
            print(f"训练集大小: {len(train_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")
            
            # 转换为numpy数组
            train_data, train_labels = self._dataset_to_numpy(train_dataset)
            test_data, test_labels = self._dataset_to_numpy(test_dataset)
            
            return train_data, train_labels, test_data, test_labels
            
        except Exception as e:
            print(f"加载数据集 {dataset_name} 时出错: {str(e)}")
            raise
    
    def _dataset_to_numpy(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        将PyTorch数据集转换为numpy数组
        
        Args:
            dataset: PyTorch数据集对象
            
        Returns:
            tuple: (data, labels)
        """
        data_list = []
        labels_list = []
        
        # 使用DataLoader批量处理数据
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        
        for batch_data, batch_labels in dataloader:
            data_list.append(batch_data.numpy())
            labels_list.append(batch_labels.numpy())
        
        # 合并所有批次
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return data, labels
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            dict: 数据集信息
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        return {
            "name": dataset_name,
            "num_classes": config["num_classes"],
            "input_shape": config["input_shape"],
            "supported": True
        }
    
    def list_supported_datasets(self) -> list:
        """
        列出所有支持的数据集
        
        Returns:
            list: 支持的数据集列表
        """
        return self.supported_datasets.copy()
    
    def check_dataset_exists(self, dataset_name: str) -> bool:
        """
        检查数据集是否已经下载到本地
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            bool: 数据集是否存在
        """
        if dataset_name not in self.supported_datasets:
            return False
        
        # 检查数据集目录是否存在
        dataset_dir_map = {
            "CIFAR-10": "cifar-10-batches-py",
            "CIFAR-100": "cifar-100-python", 
            "MNIST": "MNIST",
            "Fashion-MNIST": "FashionMNIST"
        }
        
        dataset_dir = os.path.join(self.data_dir, dataset_dir_map[dataset_name])
        return os.path.exists(dataset_dir)


class DataManager:
    """数据管理器，整合数据加载和处理功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_dir = config.get("data_dir", "./datasets")
        self.loader = DatasetLoader(self.data_dir)
    
    def load_dataset(self, dataset_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载数据集
        
        Args:
            dataset_name: 数据集名称，如果为None则使用配置文件中的设置
            
        Returns:
            tuple: (train_data, train_labels, test_data, test_labels)
        """
        if dataset_name is None:
            dataset_name = self.config.get("name", "CIFAR-10")
        
        return self.loader.load_dataset(dataset_name)
    
    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            dataset_name: 数据集名称，如果为None则使用配置文件中的设置
            
        Returns:
            dict: 数据集信息
        """
        if dataset_name is None:
            dataset_name = self.config.get("name", "CIFAR-10")
        
        return self.loader.get_dataset_info(dataset_name)