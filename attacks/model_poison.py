"""
模型投毒攻击模块
Model Poisoning Attack Module

实现各种模型投毒攻击策略，包括高斯噪声攻击、缩放攻击、IPM攻击和自定义模型投毒接口
"""

import numpy as np
import torch
import copy
from typing import Dict, Any, Callable, Optional, List
import logging


class ModelPoisonAttack:
    """
    模型投毒攻击类
    支持高斯噪声攻击、缩放攻击、IPM攻击和自定义模型投毒接口
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        """
        初始化模型投毒攻击
        
        Args:
            attack_config: 攻击配置参数
        """
        self.attack_config = attack_config
        self.attack_type = attack_config.get('attack_type', 'gaussian')
        self.attack_params = attack_config.get('attack_params', {})
        self.malicious_clients = attack_config.get('malicious_clients', [])
        
        # 验证攻击参数
        self._validate_attack_params()
        
        self.logger = logging.getLogger(__name__)
        
    def _validate_attack_params(self):
        """验证攻击参数的有效性"""
        if self.attack_type == 'gaussian':
            noise_scale = self.attack_params.get('noise_scale', 0.1)
            if noise_scale < 0:
                raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
        
        elif self.attack_type == 'scaling':
            scale_factor = self.attack_params.get('scale_factor', 10)
            if scale_factor <= 0:
                raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        
        elif self.attack_type == 'ipm':
            # IPM攻击参数验证将在具体使用时进行
            pass
        
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
    
    def _dict_to_tensor_dict(self, param_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        将参数字典转换为张量字典
        
        Args:
            param_dict: 参数字典
            
        Returns:
            Dict[str, torch.Tensor]: 张量字典
        """
        tensor_dict = {}
        for key, value in param_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_dict[key] = value
            elif isinstance(value, np.ndarray):
                tensor_dict[key] = torch.from_numpy(value)
            else:
                tensor_dict[key] = torch.tensor(value)
        return tensor_dict
    
    def _tensor_dict_to_dict(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        将张量字典转换为普通字典
        
        Args:
            tensor_dict: 张量字典
            
        Returns:
            Dict[str, Any]: 普通字典
        """
        result_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                result_dict[key] = value.detach().clone()
            else:
                result_dict[key] = value
        return result_dict
    
    def gaussian_noise(self, model_params: Dict[str, Any], 
                      noise_scale: Optional[float] = None) -> Dict[str, Any]:
        """
        高斯噪声攻击
        向模型参数添加高斯噪声
        
        Args:
            model_params: 模型参数字典
            noise_scale: 噪声标准差，如果为None则使用配置中的值
            
        Returns:
            Dict[str, Any]: 攻击后的模型参数
        """
        if noise_scale is None:
            noise_scale = self.attack_params.get('noise_scale', 0.1)
            
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
        
        # 转换为张量字典
        tensor_params = self._dict_to_tensor_dict(model_params)
        poisoned_params = {}
        
        total_params = 0
        for key, param in tensor_params.items():
            # 生成与参数形状相同的高斯噪声
            noise = torch.normal(0, noise_scale, param.shape)
            poisoned_params[key] = param + noise
            total_params += param.numel()
        
        self.logger.info(f"Gaussian noise attack: added noise (scale={noise_scale:.4f}) "
                        f"to {len(poisoned_params)} parameter tensors ({total_params} total parameters)")
        
        return self._tensor_dict_to_dict(poisoned_params)
    
    def scaling_attack(self, model_params: Dict[str, Any], 
                      scale_factor: Optional[float] = None) -> Dict[str, Any]:
        """
        缩放攻击
        将模型参数乘以缩放因子
        
        Args:
            model_params: 模型参数字典
            scale_factor: 缩放因子，如果为None则使用配置中的值
            
        Returns:
            Dict[str, Any]: 攻击后的模型参数
        """
        if scale_factor is None:
            scale_factor = self.attack_params.get('scale_factor', 10)
            
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        
        # 转换为张量字典
        tensor_params = self._dict_to_tensor_dict(model_params)
        poisoned_params = {}
        
        total_params = 0
        for key, param in tensor_params.items():
            poisoned_params[key] = param * scale_factor
            total_params += param.numel()
        
        self.logger.info(f"Scaling attack: scaled {len(poisoned_params)} parameter tensors "
                        f"by factor {scale_factor} ({total_params} total parameters)")
        
        return self._tensor_dict_to_dict(poisoned_params)
    
    def ipm_attack(self, model_params: Dict[str, Any], 
                  target_params: Dict[str, Any],
                  attack_strength: Optional[float] = None) -> Dict[str, Any]:
        """
        IPM (Inner Product Manipulation) 攻击
        通过操纵内积来影响聚合结果
        
        Args:
            model_params: 当前模型参数字典
            target_params: 目标模型参数字典
            attack_strength: 攻击强度，如果为None则使用配置中的值
            
        Returns:
            Dict[str, Any]: 攻击后的模型参数
        """
        if attack_strength is None:
            attack_strength = self.attack_params.get('attack_strength', 1.0)
            
        if attack_strength < 0:
            raise ValueError(f"attack_strength must be non-negative, got {attack_strength}")
        
        # 转换为张量字典
        current_params = self._dict_to_tensor_dict(model_params)
        target_tensor_params = self._dict_to_tensor_dict(target_params)
        
        # 验证参数键匹配
        if set(current_params.keys()) != set(target_tensor_params.keys()):
            raise ValueError("Model parameters and target parameters must have the same keys")
        
        poisoned_params = {}
        total_params = 0
        
        for key in current_params.keys():
            current_param = current_params[key]
            target_param = target_tensor_params[key]
            
            # 验证参数形状匹配
            if current_param.shape != target_param.shape:
                raise ValueError(f"Parameter {key} shape mismatch: "
                               f"{current_param.shape} vs {target_param.shape}")
            
            # IPM攻击：向目标参数方向移动
            direction = target_param - current_param
            poisoned_params[key] = current_param + attack_strength * direction
            total_params += current_param.numel()
        
        self.logger.info(f"IPM attack: manipulated {len(poisoned_params)} parameter tensors "
                        f"with strength {attack_strength} ({total_params} total parameters)")
        
        return self._tensor_dict_to_dict(poisoned_params)
    
    def custom_model_poison(self, model_params: Dict[str, Any], 
                           poison_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                           **kwargs) -> Dict[str, Any]:
        """
        自定义模型投毒接口
        允许用户定义自己的模型投毒函数
        
        Args:
            model_params: 模型参数字典
            poison_func: 自定义投毒函数，接受参数字典返回投毒后的参数字典
            **kwargs: 传递给poison_func的额外参数
            
        Returns:
            Dict[str, Any]: 攻击后的模型参数
        """
        try:
            poisoned_params = poison_func(model_params, **kwargs)
            
            # 验证返回的参数格式
            if not isinstance(poisoned_params, dict):
                raise ValueError("poison_func must return a dictionary")
            
            # 验证参数键匹配
            if set(poisoned_params.keys()) != set(model_params.keys()):
                raise ValueError("Poisoned parameters must have the same keys as original parameters")
            
            # 验证参数形状匹配
            for key in model_params.keys():
                original_shape = model_params[key].shape if hasattr(model_params[key], 'shape') else None
                poisoned_shape = poisoned_params[key].shape if hasattr(poisoned_params[key], 'shape') else None
                
                if original_shape is not None and poisoned_shape is not None:
                    if original_shape != poisoned_shape:
                        raise ValueError(f"Parameter {key} shape mismatch after poisoning: "
                                       f"{original_shape} vs {poisoned_shape}")
            
            self.logger.info(f"Custom model poisoning applied to {len(poisoned_params)} parameter tensors")
            return poisoned_params
            
        except Exception as e:
            self.logger.error(f"Custom model poisoning failed: {str(e)}")
            raise
    
    def apply_attack(self, client_id: int, model_params: Dict[str, Any], 
                    target_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        对指定客户端应用模型投毒攻击
        
        Args:
            client_id: 客户端ID
            model_params: 模型参数字典
            target_params: 目标参数字典（IPM攻击需要）
            
        Returns:
            Dict[str, Any]: 攻击后的模型参数
        """
        # 检查是否为攻击者客户端
        if not self.is_malicious_client(client_id):
            return model_params
        
        self.logger.info(f"Applying {self.attack_type} model attack to client {client_id}")
        
        # 根据攻击类型应用相应的攻击
        if self.attack_type == 'gaussian':
            return self.gaussian_noise(model_params)
        elif self.attack_type == 'scaling':
            return self.scaling_attack(model_params)
        elif self.attack_type == 'ipm':
            if target_params is None:
                self.logger.warning("IPM attack requires target_params, returning original parameters")
                return model_params
            return self.ipm_attack(model_params, target_params)
        else:
            self.logger.warning(f"Unknown attack type: {self.attack_type}, returning original parameters")
            return model_params
    
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


class ModelAttackManager:
    """
    模型攻击管理器
    负责管理和协调各种模型攻击策略
    """
    
    def __init__(self, attack_config: Dict[str, Any]):
        """
        初始化模型攻击管理器
        
        Args:
            attack_config: 攻击配置
        """
        self.attack_config = attack_config
        self.attack_enabled = attack_config.get('enable', False)
        
        if self.attack_enabled:
            self.model_poison_attack = ModelPoisonAttack(attack_config)
        else:
            self.model_poison_attack = None
            
        self.logger = logging.getLogger(__name__)
    
    def is_attack_enabled(self) -> bool:
        """检查攻击是否启用"""
        return self.attack_enabled
    
    def is_malicious_client(self, client_id: int) -> bool:
        """检查客户端是否为攻击者"""
        if not self.attack_enabled or self.model_poison_attack is None:
            return False
        return self.model_poison_attack.is_malicious_client(client_id)
    
    def apply_model_poison_attack(self, client_id: int, model_params: Dict[str, Any],
                                target_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        应用模型投毒攻击
        
        Args:
            client_id: 客户端ID
            model_params: 模型参数字典
            target_params: 目标参数字典（IPM攻击需要）
            
        Returns:
            Dict[str, Any]: 攻击后的模型参数
        """
        if not self.attack_enabled or self.model_poison_attack is None:
            return model_params
            
        return self.model_poison_attack.apply_attack(client_id, model_params, target_params)
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """
        获取攻击摘要信息
        
        Returns:
            Dict[str, Any]: 攻击摘要
        """
        if not self.attack_enabled or self.model_poison_attack is None:
            return {'attack_enabled': False}
            
        summary = {'attack_enabled': True}
        summary.update(self.model_poison_attack.get_attack_info())
        return summary