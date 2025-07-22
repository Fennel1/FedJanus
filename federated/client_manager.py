"""
联邦学习客户端管理器实现
Client manager implementation for federated learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .client import FederatedClient


class ClientManager:
    """
    联邦学习客户端管理器
    负责协调多个客户端的训练、选择和调度
    """
    
    def __init__(self, 
                 clients: List[FederatedClient],
                 selection_strategy: str = "random",
                 max_workers: Optional[int] = None,
                 device: torch.device = None):
        """
        初始化客户端管理器
        
        Args:
            clients: 客户端列表
            selection_strategy: 客户端选择策略 ("random", "all", "round_robin", "weighted")
            max_workers: 最大并行工作线程数，None表示使用默认值
            device: 计算设备
        """
        if not clients:
            raise ValueError("客户端列表不能为空")
        
        self.clients = clients
        self.num_clients = len(clients)
        self.selection_strategy = selection_strategy.lower()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 并行训练设置
        self.max_workers = max_workers if max_workers else min(4, self.num_clients)
        self.enable_parallel = True
        
        # 客户端选择状态
        self.round_robin_index = 0
        self.client_weights = None
        
        # 支持的选择策略
        self.supported_strategies = ["random", "all", "round_robin", "weighted"]
        if self.selection_strategy not in self.supported_strategies:
            raise ValueError(f"不支持的选择策略: {selection_strategy}. "
                           f"支持的策略: {self.supported_strategies}")
        
        # 训练统计
        self.training_stats = {
            'total_rounds': 0,
            'total_training_time': 0.0,
            'client_participation': {client.client_id: 0 for client in clients},
            'parallel_training_enabled': self.enable_parallel
        }
        
        # 日志设置
        self.logger = logging.getLogger('ClientManager')
        self.logger.info(f"客户端管理器初始化完成，客户端数量: {self.num_clients}, "
                        f"选择策略: {selection_strategy}, 最大并行数: {self.max_workers}")
    
    def set_client_weights(self, weights: List[float]) -> None:
        """
        设置客户端权重（用于加权选择策略）
        
        Args:
            weights: 客户端权重列表
        """
        if len(weights) != self.num_clients:
            raise ValueError(f"权重数量 ({len(weights)}) 与客户端数量 ({self.num_clients}) 不匹配")
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("权重总和必须大于0")
        
        self.client_weights = [w / total_weight for w in weights]
        self.logger.info(f"客户端权重已设置: {self.client_weights}")
    
    def select_clients(self, 
                      num_clients: int, 
                      strategy: Optional[str] = None,
                      exclude_clients: Optional[List[int]] = None) -> List[FederatedClient]:
        """
        选择参与训练的客户端
        
        Args:
            num_clients: 要选择的客户端数量
            strategy: 选择策略，None表示使用默认策略
            exclude_clients: 要排除的客户端ID列表
            
        Returns:
            选中的客户端列表
        """
        if num_clients <= 0:
            raise ValueError("选择的客户端数量必须大于0")
        
        strategy = strategy or self.selection_strategy
        exclude_clients = exclude_clients or []
        
        # 过滤可用客户端
        available_clients = [client for client in self.clients 
                           if client.client_id not in exclude_clients]
        
        if not available_clients:
            raise ValueError("没有可用的客户端")
        
        if num_clients > len(available_clients):
            self.logger.warning(f"请求的客户端数量 ({num_clients}) 超过可用数量 ({len(available_clients)})")
            num_clients = len(available_clients)
        
        # 根据策略选择客户端
        if strategy == "all":
            selected_clients = available_clients[:num_clients]
        elif strategy == "random":
            selected_clients = random.sample(available_clients, num_clients)
        elif strategy == "round_robin":
            selected_clients = self._select_round_robin(available_clients, num_clients)
        elif strategy == "weighted":
            selected_clients = self._select_weighted(available_clients, num_clients)
        else:
            raise ValueError(f"不支持的选择策略: {strategy}")
        
        # 更新参与统计
        for client in selected_clients:
            self.training_stats['client_participation'][client.client_id] += 1
        
        selected_ids = [client.client_id for client in selected_clients]
        self.logger.info(f"选择了 {len(selected_clients)} 个客户端: {selected_ids}, 策略: {strategy}")
        
        return selected_clients
    
    def _select_round_robin(self, 
                           available_clients: List[FederatedClient], 
                           num_clients: int) -> List[FederatedClient]:
        """
        轮询选择客户端
        
        Args:
            available_clients: 可用客户端列表
            num_clients: 要选择的数量
            
        Returns:
            选中的客户端列表
        """
        selected_clients = []
        num_available = len(available_clients)
        
        for i in range(num_clients):
            client_index = (self.round_robin_index + i) % num_available
            selected_clients.append(available_clients[client_index])
        
        self.round_robin_index = (self.round_robin_index + num_clients) % num_available
        return selected_clients
    
    def _select_weighted(self, 
                        available_clients: List[FederatedClient], 
                        num_clients: int) -> List[FederatedClient]:
        """
        加权选择客户端
        
        Args:
            available_clients: 可用客户端列表
            num_clients: 要选择的数量
            
        Returns:
            选中的客户端列表
        """
        if self.client_weights is None:
            self.logger.warning("未设置客户端权重，使用随机选择")
            return random.sample(available_clients, num_clients)
        
        # 获取可用客户端的权重
        available_weights = []
        for client in available_clients:
            client_index = next(i for i, c in enumerate(self.clients) if c.client_id == client.client_id)
            available_weights.append(self.client_weights[client_index])
        
        # 归一化权重
        total_weight = sum(available_weights)
        if total_weight <= 0:
            return random.sample(available_clients, num_clients)
        
        normalized_weights = [w / total_weight for w in available_weights]
        
        # 加权随机选择
        selected_indices = np.random.choice(
            len(available_clients), 
            size=num_clients, 
            replace=False, 
            p=normalized_weights
        )
        
        return [available_clients[i] for i in selected_indices]
    
    def train_clients_sequential(self, 
                               selected_clients: List[FederatedClient],
                               epochs: int = 5,
                               learning_rate: float = 0.01,
                               batch_size: int = 32,
                               round_num: int = 0) -> List[Dict[str, Any]]:
        """
        顺序训练客户端
        
        Args:
            selected_clients: 选中的客户端列表
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            round_num: 联邦学习轮次
            
        Returns:
            训练结果列表
        """
        training_results = []
        start_time = time.time()
        
        self.logger.info(f"开始顺序训练 {len(selected_clients)} 个客户端，轮次: {round_num}")
        
        for i, client in enumerate(selected_clients):
            try:
                self.logger.debug(f"训练客户端 {client.client_id} ({i+1}/{len(selected_clients)})")
                
                result = client.train(
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    round_num=round_num
                )
                
                training_results.append(result)
                
            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 训练失败: {str(e)}")
                # 创建失败结果记录
                error_result = {
                    'client_id': client.client_id,
                    'round': round_num,
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'samples': 0,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'error': str(e)
                }
                training_results.append(error_result)
        
        training_time = time.time() - start_time
        self.training_stats['total_training_time'] += training_time
        
        self.logger.info(f"顺序训练完成，耗时: {training_time:.2f}秒")
        
        return training_results
    
    def train_clients_parallel(self, 
                             selected_clients: List[FederatedClient],
                             epochs: int = 5,
                             learning_rate: float = 0.01,
                             batch_size: int = 32,
                             round_num: int = 0) -> List[Dict[str, Any]]:
        """
        并行训练客户端
        
        Args:
            selected_clients: 选中的客户端列表
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            round_num: 联邦学习轮次
            
        Returns:
            训练结果列表
        """
        training_results = []
        start_time = time.time()
        
        self.logger.info(f"开始并行训练 {len(selected_clients)} 个客户端，轮次: {round_num}, "
                        f"最大并行数: {self.max_workers}")
        
        def train_single_client(client: FederatedClient) -> Dict[str, Any]:
            """训练单个客户端的内部函数"""
            try:
                return client.train(
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    round_num=round_num
                )
            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 训练失败: {str(e)}")
                return {
                    'client_id': client.client_id,
                    'round': round_num,
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'samples': 0,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'error': str(e)
                }
        
        # 使用线程池执行并行训练
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有训练任务
            future_to_client = {
                executor.submit(train_single_client, client): client 
                for client in selected_clients
            }
            
            # 收集结果
            for future in as_completed(future_to_client):
                client = future_to_client[future]
                try:
                    result = future.result()
                    training_results.append(result)
                    self.logger.debug(f"客户端 {client.client_id} 训练完成")
                except Exception as e:
                    self.logger.error(f"客户端 {client.client_id} 训练异常: {str(e)}")
                    error_result = {
                        'client_id': client.client_id,
                        'round': round_num,
                        'loss': float('inf'),
                        'accuracy': 0.0,
                        'samples': 0,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'error': str(e)
                    }
                    training_results.append(error_result)
        
        # 按客户端ID排序结果
        training_results.sort(key=lambda x: x['client_id'])
        
        training_time = time.time() - start_time
        self.training_stats['total_training_time'] += training_time
        
        self.logger.info(f"并行训练完成，耗时: {training_time:.2f}秒")
        
        return training_results
    
    def train_clients(self, 
                     selected_clients: List[FederatedClient],
                     epochs: int = 5,
                     learning_rate: float = 0.01,
                     batch_size: int = 32,
                     round_num: int = 0,
                     parallel: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        训练客户端（自动选择并行或顺序）
        
        Args:
            selected_clients: 选中的客户端列表
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            round_num: 联邦学习轮次
            parallel: 是否并行训练，None表示使用默认设置
            
        Returns:
            训练结果列表
        """
        if not selected_clients:
            raise ValueError("选中的客户端列表不能为空")
        
        use_parallel = parallel if parallel is not None else self.enable_parallel
        
        # 更新统计
        self.training_stats['total_rounds'] += 1
        
        if use_parallel and len(selected_clients) > 1:
            return self.train_clients_parallel(
                selected_clients, epochs, learning_rate, batch_size, round_num
            )
        else:
            return self.train_clients_sequential(
                selected_clients, epochs, learning_rate, batch_size, round_num
            )
    
    def evaluate_clients(self, 
                        selected_clients: List[FederatedClient],
                        use_test_data: bool = True,
                        parallel: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        评估客户端模型
        
        Args:
            selected_clients: 要评估的客户端列表
            use_test_data: 是否使用测试数据
            parallel: 是否并行评估
            
        Returns:
            评估结果列表
        """
        if not selected_clients:
            raise ValueError("客户端列表不能为空")
        
        use_parallel = parallel if parallel is not None else self.enable_parallel
        
        def evaluate_single_client(client: FederatedClient) -> Dict[str, Any]:
            """评估单个客户端的内部函数"""
            try:
                loss, accuracy = client.evaluate(use_test_data=use_test_data)
                return {
                    'client_id': client.client_id,
                    'loss': loss,
                    'accuracy': accuracy,
                    'samples': len(client.test_dataset if use_test_data else client.train_dataset)
                }
            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 评估失败: {str(e)}")
                return {
                    'client_id': client.client_id,
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'samples': 0,
                    'error': str(e)
                }
        
        evaluation_results = []
        
        if use_parallel and len(selected_clients) > 1:
            # 并行评估
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_client = {
                    executor.submit(evaluate_single_client, client): client 
                    for client in selected_clients
                }
                
                for future in as_completed(future_to_client):
                    result = future.result()
                    evaluation_results.append(result)
        else:
            # 顺序评估
            for client in selected_clients:
                result = evaluate_single_client(client)
                evaluation_results.append(result)
        
        # 按客户端ID排序
        evaluation_results.sort(key=lambda x: x['client_id'])
        
        data_type = "测试" if use_test_data else "训练"
        self.logger.info(f"客户端{data_type}数据评估完成，评估了 {len(selected_clients)} 个客户端")
        
        return evaluation_results
    
    def get_client_models(self, 
                         selected_clients: List[FederatedClient]) -> List[Dict[str, torch.Tensor]]:
        """
        获取客户端模型参数
        
        Args:
            selected_clients: 客户端列表
            
        Returns:
            客户端模型参数列表
        """
        client_models = []
        
        for client in selected_clients:
            try:
                model_params = client.get_model_parameters()
                client_models.append(model_params)
            except Exception as e:
                self.logger.error(f"获取客户端 {client.client_id} 模型参数失败: {str(e)}")
                # 添加空参数字典作为占位符
                client_models.append({})
        
        self.logger.debug(f"获取了 {len(client_models)} 个客户端的模型参数")
        
        return client_models
    
    def broadcast_model_to_clients(self, 
                                  selected_clients: List[FederatedClient],
                                  global_model_params: Dict[str, torch.Tensor]) -> None:
        """
        向客户端广播全局模型参数
        
        Args:
            selected_clients: 客户端列表
            global_model_params: 全局模型参数
        """
        for client in selected_clients:
            try:
                client.set_model_parameters(global_model_params)
            except Exception as e:
                self.logger.error(f"向客户端 {client.client_id} 广播模型参数失败: {str(e)}")
        
        client_ids = [client.client_id for client in selected_clients]
        self.logger.info(f"已向客户端 {client_ids} 广播全局模型参数")
    
    def apply_attacks_to_clients(self, 
                               malicious_clients: List[int],
                               attack_type: str,
                               attack_params: Dict[str, Any]) -> None:
        """
        对指定客户端应用攻击
        
        Args:
            malicious_clients: 恶意客户端ID列表
            attack_type: 攻击类型
            attack_params: 攻击参数
        """
        applied_count = 0
        
        for client in self.clients:
            if client.client_id in malicious_clients:
                try:
                    client.apply_attack(attack_type, attack_params)
                    applied_count += 1
                except Exception as e:
                    self.logger.error(f"对客户端 {client.client_id} 应用攻击失败: {str(e)}")
        
        self.logger.info(f"已对 {applied_count} 个客户端应用 {attack_type} 攻击")
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """
        获取客户端管理器统计信息
        
        Returns:
            统计信息字典
        """
        # 计算客户端数据统计
        total_train_samples = sum(len(client.train_dataset) for client in self.clients)
        total_test_samples = sum(len(client.test_dataset) for client in self.clients)
        malicious_clients = [client.client_id for client in self.clients if client.is_malicious]
        
        # 计算参与度统计
        participation_stats = self.training_stats['client_participation']
        avg_participation = np.mean(list(participation_stats.values())) if participation_stats else 0
        max_participation = max(participation_stats.values()) if participation_stats else 0
        min_participation = min(participation_stats.values()) if participation_stats else 0
        
        return {
            'num_clients': self.num_clients,
            'selection_strategy': self.selection_strategy,
            'max_workers': self.max_workers,
            'parallel_enabled': self.enable_parallel,
            'total_train_samples': total_train_samples,
            'total_test_samples': total_test_samples,
            'malicious_clients': malicious_clients,
            'num_malicious': len(malicious_clients),
            'training_stats': self.training_stats.copy(),
            'participation_stats': {
                'average': avg_participation,
                'maximum': max_participation,
                'minimum': min_participation,
                'per_client': participation_stats.copy()
            }
        }
    
    def reset_training_stats(self) -> None:
        """
        重置训练统计信息
        """
        self.training_stats = {
            'total_rounds': 0,
            'total_training_time': 0.0,
            'client_participation': {client.client_id: 0 for client in self.clients},
            'parallel_training_enabled': self.enable_parallel
        }
        
        self.logger.info("训练统计信息已重置")
    
    def set_parallel_training(self, enable: bool, max_workers: Optional[int] = None) -> None:
        """
        设置并行训练配置
        
        Args:
            enable: 是否启用并行训练
            max_workers: 最大工作线程数
        """
        self.enable_parallel = enable
        
        if max_workers is not None:
            self.max_workers = max_workers
        
        self.training_stats['parallel_training_enabled'] = enable
        
        self.logger.info(f"并行训练设置已更新: 启用={enable}, 最大线程数={self.max_workers}")
    
    def get_client_by_id(self, client_id: int) -> Optional[FederatedClient]:
        """
        根据ID获取客户端
        
        Args:
            client_id: 客户端ID
            
        Returns:
            客户端对象，如果不存在则返回None
        """
        for client in self.clients:
            if client.client_id == client_id:
                return client
        return None
    
    def get_clients_by_ids(self, client_ids: List[int]) -> List[FederatedClient]:
        """
        根据ID列表获取客户端列表
        
        Args:
            client_ids: 客户端ID列表
            
        Returns:
            客户端列表
        """
        selected_clients = []
        for client_id in client_ids:
            client = self.get_client_by_id(client_id)
            if client:
                selected_clients.append(client)
            else:
                self.logger.warning(f"未找到客户端ID: {client_id}")
        
        return selected_clients
    
    def __str__(self) -> str:
        """
        返回客户端管理器的字符串表示
        """
        return (f"ClientManager(clients={self.num_clients}, "
                f"strategy={self.selection_strategy}, "
                f"parallel={self.enable_parallel}, "
                f"max_workers={self.max_workers})")
    
    def __repr__(self) -> str:
        """
        返回客户端管理器的详细字符串表示
        """
        return self.__str__()