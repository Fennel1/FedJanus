"""
联邦学习服务器实现
Server implementation for federated learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import copy
import logging
from collections import OrderedDict

from .aggregation import AggregationFactory, GradientAggregation


class FederatedServer:
    """
    联邦学习服务器类
    负责全局模型管理、客户端协调和模型聚合
    """
    
    def __init__(self, 
                 global_model: nn.Module,
                 aggregation_method: str = "fedavg",
                 device: torch.device = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化联邦学习服务器
        
        Args:
            global_model: 全局模型
            aggregation_method: 聚合方法 ("fedavg", "weighted_avg", "krum", "multi_krum")
            device: 计算设备
            config: 配置字典（用于防御策略参数）
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = global_model.to(self.device)
        self.aggregation_method = aggregation_method.lower()
        self.config = config
        
        # 支持的聚合方法
        self.supported_methods = AggregationFactory.get_supported_strategies()
        
        if self.aggregation_method not in self.supported_methods:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}. "
                           f"支持的方法: {self.supported_methods}")
        
        # 创建聚合器和梯度聚合器
        self.aggregator = AggregationFactory.create_aggregator(
            self.aggregation_method, 
            self.device, 
            config=self.config
        )
        self.gradient_aggregator = GradientAggregation(self.device)
        
        # 服务器状态
        self.current_round = 0
        self.client_updates_history = []
        self.global_model_history = []
        
        # 日志设置
        self.logger = logging.getLogger('FederatedServer')
        
        self.logger.info(f"联邦学习服务器初始化完成，聚合方法: {aggregation_method}")
    
    def get_global_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取全局模型参数
        
        Returns:
            全局模型参数字典
        """
        return {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
    
    def get_global_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取全局模型状态字典
        
        Returns:
            全局模型状态字典
        """
        return {name: param.clone().detach() for name, param in self.global_model.state_dict().items()}
    
    def set_global_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        设置全局模型参数
        
        Args:
            parameters: 模型参数字典
        """
        model_dict = self.global_model.state_dict()
        
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name].copy_(param.to(self.device))
            else:
                self.logger.warning(f"参数键 {name} 在全局模型中不存在")
    
    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        """
        广播全局模型参数给客户端
        
        Returns:
            全局模型参数字典
        """
        global_params = self.get_global_model_parameters()
        self.logger.debug(f"广播全局模型参数，轮次: {self.current_round}")
        return global_params
    
    def aggregate_models(self, 
                        client_models: List[Dict[str, torch.Tensor]], 
                        client_weights: Optional[List[float]] = None,
                        client_info: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表（用于加权聚合）
            client_info: 客户端信息列表（包含样本数量等）
            
        Returns:
            聚合后的模型参数字典
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        # 处理权重
        if client_weights is None:
            if self.aggregation_method == "weighted_avg" and client_info is not None:
                # 根据客户端样本数量计算权重
                client_weights = [info.get('samples', 1) for info in client_info]
                total_samples = sum(client_weights)
                client_weights = [w / total_samples for w in client_weights]
            else:
                # 等权重
                client_weights = [1.0 / num_clients] * num_clients
        
        # 验证权重数量
        if len(client_weights) != num_clients:
            raise ValueError(f"权重数量 ({len(client_weights)}) 与客户端数量 ({num_clients}) 不匹配")
        
        # 归一化权重
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / num_clients] * num_clients
        
        self.logger.info(f"开始聚合 {num_clients} 个客户端模型，聚合方法: {self.aggregation_method}")
        
        # 使用聚合器执行聚合
        aggregated_params = self.aggregator.aggregate(
            client_models, 
            client_weights=client_weights,
            client_info=client_info
        )
        
        # 记录聚合历史
        self.client_updates_history.append({
            'round': self.current_round,
            'num_clients': num_clients,
            'client_weights': client_weights.copy(),
            'aggregation_method': self.aggregation_method
        })
        
        self.logger.info(f"模型聚合完成，轮次: {self.current_round}")
        
        return aggregated_params
    

    
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
        return self.gradient_aggregator.aggregate_gradients(client_gradients, client_weights)
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]) -> None:
        """
        更新全局模型
        
        Args:
            aggregated_params: 聚合后的模型参数
        """
        # 保存当前全局模型状态
        current_state = self.get_global_model_state_dict()
        self.global_model_history.append({
            'round': self.current_round,
            'state_dict': current_state
        })
        
        # 更新全局模型
        self.set_global_model_parameters(aggregated_params)
        
        self.logger.info(f"全局模型已更新，轮次: {self.current_round}")
    
    def evaluate_global_model(self, 
                            test_data: np.ndarray, 
                            test_labels: np.ndarray, 
                            batch_size: int = 128) -> Tuple[float, float]:
        """
        评估全局模型性能
        
        Args:
            test_data: 测试数据
            test_labels: 测试标签
            batch_size: 批次大小
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.global_model.eval()
        
        # 转换数据
        test_data_tensor = torch.FloatTensor(test_data).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        self.logger.info(f"全局模型评估完成 - 轮次: {self.current_round}, "
                        f"损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def next_round(self) -> None:
        """
        进入下一轮联邦学习
        """
        self.current_round += 1
        self.logger.info(f"进入联邦学习轮次: {self.current_round}")
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """
        获取服务器统计信息
        
        Returns:
            服务器统计字典
        """
        return {
            'current_round': self.current_round,
            'aggregation_method': self.aggregation_method,
            'total_rounds_completed': len(self.client_updates_history),
            'global_model_parameters': sum(p.numel() for p in self.global_model.parameters()),
            'device': str(self.device),
            'supported_methods': self.supported_methods
        }
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """
        获取聚合历史记录
        
        Returns:
            聚合历史列表
        """
        return self.client_updates_history.copy()
    
    def reset_server_state(self) -> None:
        """
        重置服务器状态
        """
        self.current_round = 0
        self.client_updates_history.clear()
        self.global_model_history.clear()
        
        self.logger.info("服务器状态已重置")
    
    def save_global_model(self, filepath: str) -> None:
        """
        保存全局模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'global_model_state_dict': self.global_model.state_dict(),
            'current_round': self.current_round,
            'aggregation_method': self.aggregation_method,
            'client_updates_history': self.client_updates_history,
            'server_statistics': self.get_server_statistics()
        }, filepath)
        
        self.logger.info(f"全局模型已保存到: {filepath}")
    
    def load_global_model(self, filepath: str) -> None:
        """
        加载全局模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        self.current_round = checkpoint.get('current_round', 0)
        self.aggregation_method = checkpoint.get('aggregation_method', 'fedavg')
        self.client_updates_history = checkpoint.get('client_updates_history', [])
        
        self.logger.info(f"全局模型已从 {filepath} 加载，当前轮次: {self.current_round}")
    
    def clone_global_model(self) -> nn.Module:
        """
        克隆全局模型
        
        Returns:
            克隆的全局模型
        """
        cloned_model = copy.deepcopy(self.global_model)
        cloned_model.to(self.device)
        return cloned_model
    
    def get_model_diff(self, 
                      client_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算客户端模型与全局模型的差异
        
        Args:
            client_model: 客户端模型参数
            
        Returns:
            模型参数差异字典
        """
        global_params = self.get_global_model_parameters()
        model_diff = {}
        
        for param_name in global_params.keys():
            if param_name in client_model:
                diff = client_model[param_name] - global_params[param_name]
                model_diff[param_name] = diff
            else:
                self.logger.warning(f"客户端模型缺少参数: {param_name}")
        
        return model_diff
    
    def apply_model_diff(self, 
                        model_diff: Dict[str, torch.Tensor], 
                        learning_rate: float = 1.0) -> None:
        """
        应用模型差异到全局模型
        
        Args:
            model_diff: 模型参数差异
            learning_rate: 学习率（用于控制更新幅度）
        """
        current_params = self.get_global_model_parameters()
        updated_params = {}
        
        for param_name, param in current_params.items():
            if param_name in model_diff:
                updated_params[param_name] = param + learning_rate * model_diff[param_name]
            else:
                updated_params[param_name] = param
        
        self.set_global_model_parameters(updated_params)
        
        self.logger.info(f"已应用模型差异，学习率: {learning_rate}")
    
    def __str__(self) -> str:
        """
        返回服务器的字符串表示
        """
        return (f"FederatedServer(round={self.current_round}, "
                f"method={self.aggregation_method}, "
                f"device={self.device})")
    
    def __repr__(self) -> str:
        """
        返回服务器的详细字符串表示
        """
        return self.__str__()