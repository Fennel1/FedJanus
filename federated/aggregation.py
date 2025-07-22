"""
联邦学习聚合算法实现
Aggregation algorithms for federated learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from abc import ABC, abstractmethod


class AggregationStrategy(ABC):
    """
    聚合策略抽象基类
    Abstract base class for aggregation strategies
    """
    
    @abstractmethod
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        获取聚合策略名称
        
        Returns:
            策略名称
        """
        pass


class FedAvgAggregation(AggregationStrategy):
    """
    联邦平均聚合算法
    Federated Averaging aggregation algorithm
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化FedAvg聚合器
        
        Args:
            device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('FedAvgAggregation')
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行联邦平均聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        # 处理权重
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        # 归一化权重
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / num_clients] * num_clients
        
        # 验证权重数量
        if len(client_weights) != num_clients:
            raise ValueError(f"权重数量 ({len(client_weights)}) 与客户端数量 ({num_clients}) 不匹配")
        
        self.logger.debug(f"开始FedAvg聚合，客户端数量: {num_clients}")
        
        # 获取参数键
        param_keys = client_models[0].keys()
        
        # 验证所有客户端模型具有相同的参数键
        for i, model in enumerate(client_models[1:], 1):
            if set(model.keys()) != set(param_keys):
                raise ValueError(f"客户端 {i} 的模型参数键与客户端 0 不匹配")
        
        # 初始化聚合参数
        aggregated_params = {}
        
        for param_name in param_keys:
            # 获取所有客户端的该参数
            client_params = [model[param_name] for model in client_models]
            
            # 验证参数形状一致
            param_shape = client_params[0].shape
            for i, param in enumerate(client_params[1:], 1):
                if param.shape != param_shape:
                    raise ValueError(f"客户端 {i} 的参数 {param_name} 形状与客户端 0 不匹配")
            
            # 加权平均
            weighted_param = torch.zeros_like(client_params[0], device=self.device)
            for param, weight in zip(client_params, client_weights):
                weighted_param += param.to(self.device) * weight
            
            aggregated_params[param_name] = weighted_param
        
        self.logger.debug(f"FedAvg聚合完成")
        
        return aggregated_params
    
    def get_name(self) -> str:
        """获取策略名称"""
        return "fedavg"


class WeightedAggregation(AggregationStrategy):
    """
    加权聚合算法
    Weighted aggregation algorithm based on client sample sizes
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化加权聚合器
        
        Args:
            device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('WeightedAggregation')
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 client_info: Optional[List[Dict[str, Any]]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行加权聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表
            client_info: 客户端信息列表（包含样本数量等）
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        # 根据客户端信息计算权重
        if client_weights is None and client_info is not None:
            client_weights = [info.get('samples', 1) for info in client_info]
            total_samples = sum(client_weights)
            if total_samples > 0:
                client_weights = [w / total_samples for w in client_weights]
            else:
                client_weights = [1.0 / num_clients] * num_clients
        elif client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        self.logger.debug(f"开始加权聚合，客户端数量: {num_clients}")
        
        # 使用FedAvg的聚合逻辑
        fedavg_aggregator = FedAvgAggregation(self.device)
        return fedavg_aggregator.aggregate(client_models, client_weights)
    
    def get_name(self) -> str:
        """获取策略名称"""
        return "weighted_avg"


class GradientAggregation:
    """
    梯度聚合器
    Gradient aggregation utility
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化梯度聚合器
        
        Args:
            device: 计算设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('GradientAggregation')
    
    def aggregate_gradients(self, 
                          client_gradients: List[Dict[str, torch.Tensor]], 
                          client_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合客户端梯度
        
        Args:
            client_gradients: 客户端梯度列表
            client_weights: 客户端权重列表
            
        Returns:
            聚合后的梯度字典
        """
        if not client_gradients:
            raise ValueError("客户端梯度列表不能为空")
        
        num_clients = len(client_gradients)
        
        # 处理权重
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        # 归一化权重
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / num_clients] * num_clients
        
        self.logger.debug(f"开始梯度聚合，客户端数量: {num_clients}")
        
        # 获取梯度键
        grad_keys = client_gradients[0].keys()
        
        # 验证所有客户端梯度具有相同的键
        for i, gradients in enumerate(client_gradients[1:], 1):
            if set(gradients.keys()) != set(grad_keys):
                raise ValueError(f"客户端 {i} 的梯度键与客户端 0 不匹配")
        
        # 初始化聚合梯度
        aggregated_gradients = {}
        
        for grad_name in grad_keys:
            # 获取所有客户端的该梯度
            client_grads = [gradients[grad_name] for gradients in client_gradients]
            
            # 验证梯度形状一致
            grad_shape = client_grads[0].shape
            for i, grad in enumerate(client_grads[1:], 1):
                if grad.shape != grad_shape:
                    raise ValueError(f"客户端 {i} 的梯度 {grad_name} 形状与客户端 0 不匹配")
            
            # 加权平均
            weighted_grad = torch.zeros_like(client_grads[0], device=self.device)
            for grad, weight in zip(client_grads, client_weights):
                weighted_grad += grad.to(self.device) * weight
            
            aggregated_gradients[grad_name] = weighted_grad
        
        self.logger.debug(f"梯度聚合完成")
        
        return aggregated_gradients


class AggregationFactory:
    """
    聚合策略工厂类
    Factory class for aggregation strategies
    """
    
    _strategies = {
        'fedavg': FedAvgAggregation,
        'weighted_avg': WeightedAggregation,
        'simple_avg': FedAvgAggregation,  # 简单平均等同于FedAvg
    }
    
    # 防御策略将在运行时注册
    _defense_strategies = {}
    
    @classmethod
    def create_aggregator(cls, 
                         strategy_name: str, 
                         device: torch.device = None,
                         config: Optional[Dict[str, Any]] = None,
                         **kwargs) -> AggregationStrategy:
        """
        创建聚合策略实例
        
        Args:
            strategy_name: 策略名称
            device: 计算设备
            config: 配置字典（用于防御策略）
            **kwargs: 其他参数
            
        Returns:
            聚合策略实例
        """
        strategy_name = strategy_name.lower()
        
        # 首先检查防御策略
        if strategy_name in cls._defense_strategies:
            strategy_class = cls._defense_strategies[strategy_name]
            
            # 对于Krum防御，使用特殊的创建逻辑
            if strategy_name in ['krum', 'multi_krum']:
                from defenses.krum import create_krum_defense
                
                # 处理multi_krum参数
                if strategy_name == 'multi_krum':
                    kwargs['multi_krum'] = True
                
                # 从config中获取参数
                if config and 'defense_params' in config:
                    defense_params = config['defense_params']
                    kwargs.update(defense_params)
                
                return create_krum_defense(config=config, device=device, **kwargs)
            
            # 对于中位数防御，使用特殊的创建逻辑
            elif strategy_name in ['median', 'coordinate_median', 'client_median', 'trimmed_mean']:
                from defenses.median import create_median_defense
                
                # 处理不同的中位数策略参数
                if strategy_name == 'coordinate_median':
                    kwargs['coordinate_wise'] = True
                elif strategy_name == 'client_median':
                    kwargs['coordinate_wise'] = False
                elif strategy_name == 'trimmed_mean':
                    kwargs['trimmed_mean'] = True
                
                # 从config中获取参数
                if config and 'defense_params' in config:
                    defense_params = config['defense_params']
                    kwargs.update(defense_params)
                
                return create_median_defense(config=config, device=device, **kwargs)
            
            else:
                return strategy_class(device=device, **kwargs)
        
        # 然后检查常规聚合策略
        if strategy_name not in cls._strategies:
            all_strategies = list(cls._strategies.keys()) + list(cls._defense_strategies.keys())
            raise ValueError(f"不支持的聚合策略: {strategy_name}. "
                           f"支持的策略: {all_strategies}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(device=device, **kwargs)
    
    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        获取支持的聚合策略列表
        
        Returns:
            支持的策略名称列表
        """
        return list(cls._strategies.keys()) + list(cls._defense_strategies.keys())
    
    @classmethod
    def register_strategy(cls, 
                         name: str, 
                         strategy_class: type) -> None:
        """
        注册新的聚合策略
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        if not issubclass(strategy_class, AggregationStrategy):
            raise ValueError("策略类必须继承自AggregationStrategy")
        
        cls._strategies[name.lower()] = strategy_class
    
    @classmethod
    def register_defense_strategy(cls, 
                                 name: str, 
                                 strategy_class: type) -> None:
        """
        注册新的防御策略
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        if not issubclass(strategy_class, AggregationStrategy):
            raise ValueError("防御策略类必须继承自AggregationStrategy")
        
        cls._defense_strategies[name.lower()] = strategy_class


# 便利函数
def create_aggregator(strategy_name: str, 
                     device: torch.device = None,
                     config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> AggregationStrategy:
    """
    创建聚合器的便利函数
    
    Args:
        strategy_name: 策略名称
        device: 计算设备
        config: 配置字典
        **kwargs: 其他参数
        
    Returns:
        聚合策略实例
    """
    return AggregationFactory.create_aggregator(strategy_name, device, config, **kwargs)


def get_supported_aggregation_methods() -> List[str]:
    """
    获取支持的聚合方法列表
    
    Returns:
        支持的方法名称列表
    """
    return AggregationFactory.get_supported_strategies()


# 注册防御策略
def register_defense_strategies():
    """
    注册所有可用的防御策略
    """
    try:
        from defenses.krum import KrumDefense
        AggregationFactory.register_defense_strategy('krum', KrumDefense)
        AggregationFactory.register_defense_strategy('multi_krum', KrumDefense)
    except ImportError as e:
        logging.getLogger('AggregationFactory').warning(f"无法导入Krum防御策略: {e}")
    
    try:
        from defenses.median import MedianDefense
        AggregationFactory.register_defense_strategy('median', MedianDefense)
        AggregationFactory.register_defense_strategy('coordinate_median', MedianDefense)
        AggregationFactory.register_defense_strategy('client_median', MedianDefense)
        AggregationFactory.register_defense_strategy('trimmed_mean', MedianDefense)
    except ImportError as e:
        logging.getLogger('AggregationFactory').warning(f"无法导入中位数防御策略: {e}")
    
    # 注册自定义防御策略支持
    try:
        from defenses.custom_defense import CustomDefenseRegistry
        
        # 将自定义防御策略集成到AggregationFactory中
        def register_custom_strategies():
            custom_defenses = CustomDefenseRegistry.get_registered_defenses()
            for defense_name in custom_defenses:
                AggregationFactory.register_defense_strategy(defense_name, None)
        
        # 扩展AggregationFactory的create_aggregator方法以支持自定义防御
        original_create_aggregator = AggregationFactory.create_aggregator
        
        @classmethod
        def enhanced_create_aggregator(cls, 
                                     strategy_name: str, 
                                     device: torch.device = None,
                                     config: Optional[Dict[str, Any]] = None,
                                     **kwargs) -> AggregationStrategy:
            strategy_name_lower = strategy_name.lower()
            
            # 首先检查是否是自定义防御策略
            if strategy_name_lower in CustomDefenseRegistry.get_registered_defenses():
                return CustomDefenseRegistry.create_defense(
                    strategy_name_lower, 
                    config=config, 
                    device=device, 
                    **kwargs
                )
            
            # 否则使用原始的创建逻辑
            return original_create_aggregator(strategy_name, device, config, **kwargs)
        
        # 替换方法
        AggregationFactory.create_aggregator = enhanced_create_aggregator
        
        # 扩展get_supported_strategies方法
        original_get_supported_strategies = AggregationFactory.get_supported_strategies
        
        @classmethod
        def enhanced_get_supported_strategies(cls) -> List[str]:
            base_strategies = original_get_supported_strategies()
            custom_strategies = CustomDefenseRegistry.get_registered_defenses()
            return base_strategies + custom_strategies
        
        AggregationFactory.get_supported_strategies = enhanced_get_supported_strategies
        
        # 初始注册
        register_custom_strategies()
        
    except ImportError as e:
        logging.getLogger('AggregationFactory').warning(f"无法导入自定义防御策略支持: {e}")


# 自动注册防御策略
register_defense_strategies()