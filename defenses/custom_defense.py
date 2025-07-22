"""
自定义防御接口实现
Custom defense interface implementation for federated learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np
import logging
import inspect
import sys
import os
from abc import ABC, abstractmethod

# Add the parent directory to the path to import from federated module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.aggregation import AggregationStrategy


class CustomDefenseBase(AggregationStrategy):
    """
    自定义防御策略基类
    Base class for custom defense strategies
    
    所有自定义防御策略都应该继承此类并实现必要的方法。
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[Dict[str, Any]] = None,
                 device: torch.device = None):
        """
        初始化自定义防御策略
        
        Args:
            name: 防御策略名称
            config: 配置字典
            device: 计算设备
        """
        self.name = name
        self.config = config or {}
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(f'CustomDefense_{name}')
        
        # 从配置中提取防御参数
        self.defense_params = self.config.get('defense_params', {})
        
        # 验证配置
        self._validate_config()
        
        # 初始化防御策略
        self._initialize_defense()
    
    @abstractmethod
    def _initialize_defense(self) -> None:
        """
        初始化防御策略的具体实现
        子类必须实现此方法来设置特定的防御参数和状态
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        验证配置参数的有效性
        子类必须实现此方法来验证特定的配置参数
        """
        pass
    
    @abstractmethod
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行防御聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        pass
    
    def get_name(self) -> str:
        """获取防御策略名称"""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置参数
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        self.defense_params = self.config.get('defense_params', {})
        self._validate_config()
        self._initialize_defense()
    
    def get_defense_params(self) -> Dict[str, Any]:
        """获取防御参数"""
        return self.defense_params.copy()
    
    def set_defense_param(self, param_name: str, param_value: Any) -> None:
        """
        设置单个防御参数
        
        Args:
            param_name: 参数名称
            param_value: 参数值
        """
        self.defense_params[param_name] = param_value
        if 'defense_params' not in self.config:
            self.config['defense_params'] = {}
        self.config['defense_params'][param_name] = param_value
        
        # 重新验证和初始化
        self._validate_config()
        self._initialize_defense()
    
    def _compute_distance_matrix(self, client_models: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        计算客户端模型之间的欧几里得距离矩阵
        这是一个通用的工具方法，子类可以使用
        
        Args:
            client_models: 客户端模型参数列表
            
        Returns:
            距离矩阵 (num_clients x num_clients)
        """
        num_clients = len(client_models)
        distance_matrix = torch.zeros(num_clients, num_clients, device=self.device)
        
        # 验证所有客户端模型具有相同的参数键
        param_keys = set(client_models[0].keys())
        for i, model in enumerate(client_models[1:], 1):
            if set(model.keys()) != param_keys:
                raise ValueError(f"客户端 {i} 的模型参数键与客户端 0 不匹配")
        
        # 将每个客户端的模型参数展平为向量
        client_vectors = []
        for model in client_models:
            param_vector = torch.cat([param.flatten() for param in model.values()])
            client_vectors.append(param_vector.to(self.device))
        
        # 计算两两之间的欧几里得距离
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distance = torch.norm(client_vectors[i] - client_vectors[j], p=2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _average_models(self, models: List[Dict[str, torch.Tensor]], 
                       weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        对多个模型进行加权平均
        这是一个通用的工具方法，子类可以使用
        
        Args:
            models: 模型参数列表
            weights: 权重列表
            
        Returns:
            平均后的模型参数
        """
        if not models:
            raise ValueError("模型列表不能为空")
        
        num_models = len(models)
        
        # 处理权重
        if weights is None:
            weights = [1.0 / num_models] * num_models
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / num_models] * num_models
        
        param_keys = models[0].keys()
        
        # 验证所有模型具有相同的参数键
        for i, model in enumerate(models[1:], 1):
            if set(model.keys()) != set(param_keys):
                raise ValueError(f"模型 {i} 的参数键与模型 0 不匹配")
        
        # 初始化平均参数
        averaged_params = {}
        
        for param_name in param_keys:
            # 获取所有模型的该参数
            params = [model[param_name] for model in models]
            
            # 验证参数形状一致
            param_shape = params[0].shape
            for i, param in enumerate(params[1:], 1):
                if param.shape != param_shape:
                    raise ValueError(f"模型 {i} 的参数 {param_name} 形状与模型 0 不匹配")
            
            # 计算加权平均值
            averaged_param = torch.zeros_like(params[0], device=self.device)
            for param, weight in zip(params, weights):
                averaged_param += param.to(self.device) * weight
            
            averaged_params[param_name] = averaged_param
        
        return averaged_params


class FunctionBasedDefense(CustomDefenseBase):
    """
    基于函数的防御策略
    Function-based defense strategy that wraps a user-defined function
    """
    
    def __init__(self, 
                 name: str,
                 defense_function: Callable,
                 config: Optional[Dict[str, Any]] = None,
                 device: torch.device = None):
        """
        初始化基于函数的防御策略
        
        Args:
            name: 防御策略名称
            defense_function: 防御函数
            config: 配置字典
            device: 计算设备
        """
        self.defense_function = defense_function
        super().__init__(name, config, device)
    
    def _initialize_defense(self) -> None:
        """初始化防御策略"""
        # 检查防御函数的签名
        sig = inspect.signature(self.defense_function)
        self.function_params = list(sig.parameters.keys())
        
        self.logger.debug(f"防御函数参数: {self.function_params}")
    
    def _validate_config(self) -> None:
        """验证配置参数"""
        if not callable(self.defense_function):
            raise ValueError("defense_function 必须是可调用对象")
        
        # 检查函数签名是否包含必需的参数
        sig = inspect.signature(self.defense_function)
        params = list(sig.parameters.keys())
        
        if 'client_models' not in params:
            raise ValueError("防御函数必须包含 'client_models' 参数")
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行基于函数的防御聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        self.logger.debug(f"开始执行自定义防御函数: {self.name}")
        
        # 准备函数参数
        function_kwargs = {
            'client_models': client_models,
            'client_weights': client_weights,
            'device': self.device,
            **self.defense_params,
            **kwargs
        }
        
        # 过滤掉函数不需要的参数
        sig = inspect.signature(self.defense_function)
        filtered_kwargs = {}
        for param_name in sig.parameters.keys():
            if param_name in function_kwargs:
                filtered_kwargs[param_name] = function_kwargs[param_name]
        
        try:
            # 调用防御函数
            result = self.defense_function(**filtered_kwargs)
            
            # 验证返回结果
            if not isinstance(result, dict):
                raise ValueError("防御函数必须返回模型参数字典")
            
            # 验证返回的参数是否为torch.Tensor
            for param_name, param_value in result.items():
                if not isinstance(param_value, torch.Tensor):
                    raise ValueError(f"参数 {param_name} 必须是 torch.Tensor 类型")
            
            self.logger.debug(f"自定义防御函数执行完成")
            
            return result
            
        except Exception as e:
            self.logger.error(f"自定义防御函数执行失败: {e}")
            raise


class CustomDefenseRegistry:
    """
    自定义防御策略注册器
    Registry for custom defense strategies
    """
    
    _registered_defenses: Dict[str, Dict[str, Any]] = {}
    _logger = logging.getLogger('CustomDefenseRegistry')
    
    @classmethod
    def register_defense(cls, 
                        name: str, 
                        defense_class: type = None,
                        defense_function: Callable = None,
                        config_validator: Optional[Callable] = None,
                        description: str = "") -> None:
        """
        注册自定义防御策略
        
        Args:
            name: 防御策略名称
            defense_class: 防御策略类（继承自CustomDefenseBase）
            defense_function: 防御函数（用于FunctionBasedDefense）
            config_validator: 配置验证函数
            description: 防御策略描述
        """
        name = name.lower()
        
        if defense_class is not None and defense_function is not None:
            raise ValueError("不能同时指定 defense_class 和 defense_function")
        
        if defense_class is None and defense_function is None:
            raise ValueError("必须指定 defense_class 或 defense_function 之一")
        
        # 验证防御类
        if defense_class is not None:
            if not issubclass(defense_class, CustomDefenseBase):
                raise ValueError("defense_class 必须继承自 CustomDefenseBase")
        
        # 验证防御函数
        if defense_function is not None:
            if not callable(defense_function):
                raise ValueError("defense_function 必须是可调用对象")
            
            # 检查函数签名
            sig = inspect.signature(defense_function)
            params = list(sig.parameters.keys())
            if 'client_models' not in params:
                raise ValueError("防御函数必须包含 'client_models' 参数")
        
        # 验证配置验证器
        if config_validator is not None and not callable(config_validator):
            raise ValueError("config_validator 必须是可调用对象")
        
        # 注册防御策略
        cls._registered_defenses[name] = {
            'defense_class': defense_class,
            'defense_function': defense_function,
            'config_validator': config_validator,
            'description': description
        }
        
        cls._logger.info(f"注册自定义防御策略: {name}")
    
    @classmethod
    def unregister_defense(cls, name: str) -> bool:
        """
        注销自定义防御策略
        
        Args:
            name: 防御策略名称
            
        Returns:
            是否成功注销
        """
        name = name.lower()
        if name in cls._registered_defenses:
            del cls._registered_defenses[name]
            cls._logger.info(f"注销自定义防御策略: {name}")
            return True
        return False
    
    @classmethod
    def get_registered_defenses(cls) -> List[str]:
        """
        获取已注册的防御策略列表
        
        Returns:
            防御策略名称列表
        """
        return list(cls._registered_defenses.keys())
    
    @classmethod
    def get_defense_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        获取防御策略信息
        
        Args:
            name: 防御策略名称
            
        Returns:
            防御策略信息字典，如果不存在则返回None
        """
        name = name.lower()
        return cls._registered_defenses.get(name)
    
    @classmethod
    def create_defense(cls, 
                      name: str, 
                      config: Optional[Dict[str, Any]] = None,
                      device: torch.device = None,
                      **kwargs) -> CustomDefenseBase:
        """
        创建自定义防御策略实例
        
        Args:
            name: 防御策略名称
            config: 配置字典
            device: 计算设备
            **kwargs: 其他参数
            
        Returns:
            防御策略实例
        """
        name = name.lower()
        
        if name not in cls._registered_defenses:
            available_defenses = cls.get_registered_defenses()
            raise ValueError(f"未注册的防御策略: {name}. "
                           f"可用的策略: {available_defenses}")
        
        defense_info = cls._registered_defenses[name]
        
        # 验证配置
        if config is not None and defense_info['config_validator'] is not None:
            try:
                is_valid = defense_info['config_validator'](config)
                if not is_valid:
                    raise ValueError(f"防御策略 {name} 的配置验证失败")
            except Exception as e:
                cls._logger.error(f"配置验证失败: {e}")
                raise
        
        # 创建防御策略实例
        if defense_info['defense_class'] is not None:
            # 使用自定义防御类
            defense_class = defense_info['defense_class']
            return defense_class(name=name, config=config, device=device, **kwargs)
        
        elif defense_info['defense_function'] is not None:
            # 使用基于函数的防御
            defense_function = defense_info['defense_function']
            return FunctionBasedDefense(
                name=name,
                defense_function=defense_function,
                config=config,
                device=device
            )
        
        else:
            raise ValueError(f"防御策略 {name} 配置错误：缺少 defense_class 或 defense_function")
    
    @classmethod
    def validate_defense_config(cls, name: str, config: Dict[str, Any]) -> bool:
        """
        验证防御策略配置
        
        Args:
            name: 防御策略名称
            config: 配置字典
            
        Returns:
            配置是否有效
        """
        name = name.lower()
        
        if name not in cls._registered_defenses:
            cls._logger.error(f"未注册的防御策略: {name}")
            return False
        
        defense_info = cls._registered_defenses[name]
        
        if defense_info['config_validator'] is not None:
            try:
                return defense_info['config_validator'](config)
            except Exception as e:
                cls._logger.error(f"配置验证失败: {e}")
                return False
        
        # 如果没有配置验证器，认为配置有效
        return True
    
    @classmethod
    def list_defenses(cls) -> Dict[str, str]:
        """
        列出所有已注册的防御策略及其描述
        
        Returns:
            防御策略名称到描述的映射
        """
        return {name: info['description'] for name, info in cls._registered_defenses.items()}


# 便利函数
def register_custom_defense(name: str, 
                           defense_class: type = None,
                           defense_function: Callable = None,
                           config_validator: Optional[Callable] = None,
                           description: str = "") -> None:
    """
    注册自定义防御策略的便利函数
    
    Args:
        name: 防御策略名称
        defense_class: 防御策略类
        defense_function: 防御函数
        config_validator: 配置验证函数
        description: 描述
    """
    CustomDefenseRegistry.register_defense(
        name=name,
        defense_class=defense_class,
        defense_function=defense_function,
        config_validator=config_validator,
        description=description
    )


def register_function_defense(name: str, 
                             defense_function: Callable,
                             config_validator: Optional[Callable] = None,
                             description: str = "") -> None:
    """
    注册基于函数的防御策略的便利函数
    
    Args:
        name: 防御策略名称
        defense_function: 防御函数
        config_validator: 配置验证函数
        description: 描述
    """
    register_custom_defense(
        name=name,
        defense_function=defense_function,
        config_validator=config_validator,
        description=description
    )


def create_custom_defense(name: str, 
                         config: Optional[Dict[str, Any]] = None,
                         device: torch.device = None,
                         **kwargs) -> CustomDefenseBase:
    """
    创建自定义防御策略的便利函数
    
    Args:
        name: 防御策略名称
        config: 配置字典
        device: 计算设备
        **kwargs: 其他参数
        
    Returns:
        防御策略实例
    """
    return CustomDefenseRegistry.create_defense(name, config, device, **kwargs)


def get_available_custom_defenses() -> List[str]:
    """
    获取可用的自定义防御策略列表
    
    Returns:
        防御策略名称列表
    """
    return CustomDefenseRegistry.get_registered_defenses()


def validate_custom_defense_config(name: str, config: Dict[str, Any]) -> bool:
    """
    验证自定义防御策略配置
    
    Args:
        name: 防御策略名称
        config: 配置字典
        
    Returns:
        配置是否有效
    """
    return CustomDefenseRegistry.validate_defense_config(name, config)