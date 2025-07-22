"""
数据投毒攻击模块
Data Poisoning Attack Module

实现各种数据投毒攻击策略，包括标签反转攻击和自定义数据投毒接口
"""

import numpy as np
import torch
import random
from typing import Tuple, Dict, Any, Callable, Optional, List
import logging


class DataPoisonAttack:
    """
    数据投毒攻击类
    支持标签反转攻击和自定义数据投毒接口
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        """
        初始化数据投毒攻击
        
        Args:
            attack_config: 攻击配置参数
        """
        self.attack_config = attack_config
        self.attack_type = attack_config.get('attack_type', 'label_flipping')
        self.attack_params = attack_config.get('attack_params', {})
        self.malicious_clients = attack_config.get('malicious_clients', [])
        
        # 验证攻击参数
        self._validate_attack_params()
        
        self.logger = logging.getLogger(__name__)
        
    def _validate_attack_params(self):
        """验证攻击参数的有效性"""
        if self.attack_type == 'label_flipping':
            flip_ratio = self.attack_params.get('flip_ratio', 0.1)
            if not 0 <= flip_ratio <= 1:
                raise ValueError(f"flip_ratio must be between 0 and 1, got {flip_ratio}")
        
        if not isinstance(self.malicious_clients, list):
            raise ValueError("malicious_clients must be a list")
            
        for client_id in self.malicious_clients:
            if not isinstance(client_id, int) or client_id < 0:
                raise ValueError(f"Invalid malicious client ID: {client_id}")
    
    def is_malicious_client(self, client_id: int) -> bool:
        """
        检查客户端是否为攻击者
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: 是否为攻击者客户端
        """
        return client_id in self.malicious_clients
    
    def label_flipping(self, data: np.ndarray, labels: np.ndarray, 
                      flip_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        标签反转攻击
        随机反转指定比例的标签
        
        Args:
            data: 训练数据
            labels: 原始标签
            flip_ratio: 反转比例，如果为None则使用配置中的值
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 攻击后的数据和标签
        """
        if flip_ratio is None:
            flip_ratio = self.attack_params.get('flip_ratio', 0.1)
            
        # 验证flip_ratio参数
        if not 0 <= flip_ratio <= 1:
            raise ValueError(f"flip_ratio must be between 0 and 1, got {flip_ratio}")
        
        # 复制数据和标签，避免修改原始数据
        poisoned_data = data.copy()
        poisoned_labels = labels.copy()
        
        # 计算需要反转的样本数量
        num_samples = len(labels)
        num_flip = int(num_samples * flip_ratio)
        
        if num_flip > 0:
            # 随机选择要反转的样本索引
            flip_indices = np.random.choice(num_samples, num_flip, replace=False)
            
            # 获取唯一标签值
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            
            # 对选中的样本进行标签反转
            for idx in flip_indices:
                original_label = poisoned_labels[idx]
                # 随机选择一个不同的标签
                possible_labels = [label for label in unique_labels if label != original_label]
                if possible_labels:
                    poisoned_labels[idx] = np.random.choice(possible_labels)
            
            self.logger.info(f"Label flipping attack: flipped {num_flip}/{num_samples} labels "
                           f"(ratio: {flip_ratio:.2f})")
        
        return poisoned_data, poisoned_labels
    
    def custom_data_poison(self, data: np.ndarray, labels: np.ndarray, 
                          poison_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        自定义数据投毒接口
        允许用户定义自己的数据投毒函数
        
        Args:
            data: 训练数据
            labels: 原始标签
            poison_func: 自定义投毒函数，接受(data, labels)返回(poisoned_data, poisoned_labels)
            **kwargs: 传递给poison_func的额外参数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 攻击后的数据和标签
        """
        try:
            poisoned_data, poisoned_labels = poison_func(data, labels, **kwargs)
            
            # 验证返回的数据格式
            if not isinstance(poisoned_data, np.ndarray) or not isinstance(poisoned_labels, np.ndarray):
                raise ValueError("poison_func must return numpy arrays")
                
            if len(poisoned_data) != len(poisoned_labels):
                raise ValueError("Poisoned data and labels must have the same length")
                
            self.logger.info(f"Custom data poisoning applied to {len(poisoned_data)} samples")
            return poisoned_data, poisoned_labels
            
        except Exception as e:
            self.logger.error(f"Custom data poisoning failed: {str(e)}")
            raise
    
    def apply_attack(self, client_id: int, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对指定客户端应用数据投毒攻击
        
        Args:
            client_id: 客户端ID
            data: 训练数据
            labels: 原始标签
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 攻击后的数据和标签
        """
        # 检查是否为攻击者客户端
        if not self.is_malicious_client(client_id):
            return data, labels
        
        self.logger.info(f"Applying {self.attack_type} attack to client {client_id}")
        
        # 根据攻击类型应用相应的攻击
        if self.attack_type == 'label_flipping':
            return self.label_flipping(data, labels)
        else:
            self.logger.warning(f"Unknown attack type: {self.attack_type}, returning original data")
            return data, labels
    
    def get_attack_info(self) -> Dict[str, Any]:
        """
        获取攻击信息
        
        Returns:
            Dict[str, Any]: 攻击配置信息
        """
        return {
            'attack_type': self.attack_type,
            'attack_params': self.attack_params,
            'malicious_clients': self.malicious_clients,
            'num_malicious_clients': len(self.malicious_clients)
        }


class AttackManager:
    """
    攻击管理器
    负责管理和协调各种攻击策略
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        """
        初始化攻击管理器
        
        Args:
            attack_config: 攻击配置
        """
        self.attack_config = attack_config
        self.attack_enabled = attack_config.get('enable', False)
        
        if self.attack_enabled:
            self.data_poison_attack = DataPoisonAttack(attack_config)
        else:
            self.data_poison_attack = None
            
        self.logger = logging.getLogger(__name__)
    
    def is_attack_enabled(self) -> bool:
        """检查攻击是否启用"""
        return self.attack_enabled
    
    def is_malicious_client(self, client_id: int) -> bool:
        """检查客户端是否为攻击者"""
        if not self.attack_enabled or self.data_poison_attack is None:
            return False
        return self.data_poison_attack.is_malicious_client(client_id)
    
    def apply_data_poison_attack(self, client_id: int, data: np.ndarray, 
                               labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据投毒攻击
        
        Args:
            client_id: 客户端ID
            data: 训练数据
            labels: 原始标签
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 攻击后的数据和标签
        """
        if not self.attack_enabled or self.data_poison_attack is None:
            return data, labels
            
        return self.data_poison_attack.apply_attack(client_id, data, labels)
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """
        获取攻击摘要信息
        
        Returns:
            Dict[str, Any]: 攻击摘要
        """
        if not self.attack_enabled or self.data_poison_attack is None:
            return {'attack_enabled': False}
            
        summary = {'attack_enabled': True}
        summary.update(self.data_poison_attack.get_attack_info())
        return summary