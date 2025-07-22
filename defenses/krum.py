"""
Krum防御算法实现
Krum defense algorithm implementation for federated learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
import sys
import os

# Add the parent directory to the path to import from federated module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.aggregation import AggregationStrategy


class KrumDefense(AggregationStrategy):
    """
    Krum防御聚合算法
    Krum defense aggregation algorithm
    
    Krum算法通过计算客户端模型参数之间的距离，选择最可信的客户端模型进行聚合，
    从而抵御恶意客户端的投毒攻击。
    """
    
    def __init__(self, 
                 num_malicious: int = 0,
                 multi_krum: bool = False,
                 device: torch.device = None):
        """
        初始化Krum防御器
        
        Args:
            num_malicious: 预期的恶意客户端数量
            multi_krum: 是否使用Multi-Krum（选择多个客户端）
            device: 计算设备
        """
        self.num_malicious = num_malicious
        self.multi_krum = multi_krum
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('KrumDefense')
        
        # 验证参数
        if self.num_malicious < 0:
            raise ValueError("恶意客户端数量不能为负数")
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行Krum防御聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表（Krum中不使用）
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        # 验证客户端数量是否足够
        if num_clients <= 2 * self.num_malicious:
            raise ValueError(f"客户端数量 ({num_clients}) 必须大于 2 * 恶意客户端数量 ({2 * self.num_malicious})")
        
        self.logger.debug(f"开始Krum防御聚合，客户端数量: {num_clients}, 预期恶意客户端: {self.num_malicious}")
        
        # 计算客户端之间的距离矩阵
        distance_matrix = self._compute_distance_matrix(client_models)
        
        # 计算每个客户端的Krum分数
        krum_scores = self._compute_krum_scores(distance_matrix, num_clients)
        
        # 选择可信的客户端
        selected_clients = self._select_clients(krum_scores, num_clients)
        
        # 聚合选中的客户端模型
        if self.multi_krum and len(selected_clients) > 1:
            # Multi-Krum: 对选中的多个客户端进行平均
            selected_models = [client_models[i] for i in selected_clients]
            aggregated_params = self._average_models(selected_models)
            self.logger.info(f"Multi-Krum选择了 {len(selected_clients)} 个客户端进行聚合")
        else:
            # 标准Krum: 只选择最可信的一个客户端
            best_client_idx = selected_clients[0]
            aggregated_params = {k: v.clone().to(self.device) for k, v in client_models[best_client_idx].items()}
            self.logger.info(f"Krum选择了客户端 {best_client_idx} 作为全局模型")
        
        self.logger.debug(f"Krum防御聚合完成")
        
        return aggregated_params
    
    def _compute_distance_matrix(self, client_models: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        计算客户端模型之间的欧几里得距离矩阵
        
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
        
        # 验证所有客户端模型具有相同的参数形状
        for param_name in param_keys:
            reference_shape = client_models[0][param_name].shape
            for i, model in enumerate(client_models[1:], 1):
                if model[param_name].shape != reference_shape:
                    raise ValueError(f"客户端 {i} 的参数 {param_name} 形状与客户端 0 不匹配")
        
        # 将每个客户端的模型参数展平为向量
        client_vectors = []
        for model in client_models:
            # 将所有参数连接成一个向量
            param_vector = torch.cat([param.flatten() for param in model.values()])
            client_vectors.append(param_vector.to(self.device))
        
        # 计算两两之间的欧几里得距离
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distance = torch.norm(client_vectors[i] - client_vectors[j], p=2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # 对称矩阵
        
        return distance_matrix
    
    def _compute_krum_scores(self, distance_matrix: torch.Tensor, num_clients: int) -> List[float]:
        """
        计算每个客户端的Krum分数
        
        Args:
            distance_matrix: 距离矩阵
            num_clients: 客户端数量
            
        Returns:
            每个客户端的Krum分数列表
        """
        krum_scores = []
        
        # 计算需要选择的最近邻数量
        # Krum选择 n - f - 2 个最近的邻居，其中 n 是总客户端数，f 是恶意客户端数
        num_neighbors = num_clients - self.num_malicious - 2
        
        if num_neighbors <= 0:
            raise ValueError(f"无法计算Krum分数：需要的邻居数量 ({num_neighbors}) <= 0")
        
        for i in range(num_clients):
            # 获取客户端i到其他所有客户端的距离
            distances_to_i = distance_matrix[i]
            
            # 排除自己（距离为0）
            distances_to_others = torch.cat([distances_to_i[:i], distances_to_i[i+1:]])
            
            # 选择最近的 num_neighbors 个邻居
            closest_distances, _ = torch.topk(distances_to_others, 
                                            min(num_neighbors, len(distances_to_others)), 
                                            largest=False)
            
            # Krum分数是最近邻居距离的平方和
            krum_score = torch.sum(closest_distances ** 2).item()
            krum_scores.append(krum_score)
        
        return krum_scores
    
    def _select_clients(self, krum_scores: List[float], num_clients: int) -> List[int]:
        """
        根据Krum分数选择可信的客户端
        
        Args:
            krum_scores: 每个客户端的Krum分数
            num_clients: 客户端总数
            
        Returns:
            选中的客户端索引列表
        """
        # 将分数和索引配对并排序（分数越小越好）
        score_index_pairs = [(score, idx) for idx, score in enumerate(krum_scores)]
        score_index_pairs.sort(key=lambda x: x[0])
        
        if self.multi_krum:
            # Multi-Krum: 选择多个可信客户端
            num_selected = max(1, num_clients - self.num_malicious)
            selected_clients = [idx for _, idx in score_index_pairs[:num_selected]]
        else:
            # 标准Krum: 只选择分数最低的客户端
            selected_clients = [score_index_pairs[0][1]]
        
        self.logger.debug(f"Krum分数: {krum_scores}")
        self.logger.debug(f"选中的客户端: {selected_clients}")
        
        return selected_clients
    
    def _average_models(self, models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对多个模型进行简单平均
        
        Args:
            models: 模型参数列表
            
        Returns:
            平均后的模型参数
        """
        if not models:
            raise ValueError("模型列表不能为空")
        
        num_models = len(models)
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
            
            # 计算平均值
            averaged_param = torch.zeros_like(params[0], device=self.device)
            for param in params:
                averaged_param += param.to(self.device)
            averaged_param /= num_models
            
            averaged_params[param_name] = averaged_param
        
        return averaged_params
    
    def get_name(self) -> str:
        """获取策略名称"""
        return "krum" if not self.multi_krum else "multi_krum"
    
    def detect_malicious_clients(self, 
                                client_models: List[Dict[str, torch.Tensor]],
                                threshold_percentile: float = 90.0) -> List[int]:
        """
        检测可能的恶意客户端
        
        Args:
            client_models: 客户端模型参数列表
            threshold_percentile: 分数阈值百分位数
            
        Returns:
            可能恶意的客户端索引列表
        """
        if not client_models:
            return []
        
        num_clients = len(client_models)
        
        if num_clients <= 2 * self.num_malicious:
            self.logger.warning("客户端数量不足，无法进行恶意客户端检测")
            return []
        
        # 计算距离矩阵和Krum分数
        distance_matrix = self._compute_distance_matrix(client_models)
        krum_scores = self._compute_krum_scores(distance_matrix, num_clients)
        
        # 使用百分位数作为阈值
        threshold = np.percentile(krum_scores, threshold_percentile)
        
        # 分数高于阈值的客户端被认为是可疑的
        suspicious_clients = [i for i, score in enumerate(krum_scores) if score > threshold]
        
        self.logger.info(f"检测到 {len(suspicious_clients)} 个可疑客户端: {suspicious_clients}")
        self.logger.debug(f"Krum分数阈值: {threshold:.4f}")
        
        return suspicious_clients


def create_krum_defense(config: Dict[str, Any] = None, **kwargs) -> KrumDefense:
    """
    根据配置创建Krum防御实例
    
    Args:
        config: 防御配置字典
        **kwargs: 直接传递的参数（优先级高于config）
        
    Returns:
        Krum防御实例
    """
    # 首先从config获取默认参数
    num_malicious = 0
    multi_krum = False
    
    if config is not None:
        defense_params = config.get('defense_params', {})
        num_malicious = defense_params.get('num_malicious', num_malicious)
        multi_krum = defense_params.get('multi_krum', multi_krum)
    
    # kwargs参数优先级更高，会覆盖config中的参数
    num_malicious = kwargs.get('num_malicious', num_malicious)
    multi_krum = kwargs.get('multi_krum', multi_krum)
    
    # 验证参数
    if not isinstance(num_malicious, int) or num_malicious < 0:
        raise ValueError(f"num_malicious 必须是非负整数，得到: {num_malicious}")
    
    if not isinstance(multi_krum, bool):
        raise ValueError(f"multi_krum 必须是布尔值，得到: {multi_krum}")
    
    return KrumDefense(
        num_malicious=num_malicious,
        multi_krum=multi_krum,
        device=kwargs.get('device')
    )


def validate_krum_config(config: Dict[str, Any]) -> bool:
    """
    验证Krum防御配置的有效性
    
    Args:
        config: 防御配置字典
        
    Returns:
        配置是否有效
    """
    try:
        defense_params = config.get('defense_params', {})
        
        # 检查必需参数
        if 'num_malicious' not in defense_params:
            raise ValueError("缺少必需参数: num_malicious")
        
        num_malicious = defense_params['num_malicious']
        if not isinstance(num_malicious, int) or num_malicious < 0:
            raise ValueError(f"num_malicious 必须是非负整数，得到: {num_malicious}")
        
        # 检查可选参数
        if 'multi_krum' in defense_params:
            multi_krum = defense_params['multi_krum']
            if not isinstance(multi_krum, bool):
                raise ValueError(f"multi_krum 必须是布尔值，得到: {multi_krum}")
        
        return True
        
    except Exception as e:
        logging.getLogger('KrumDefense').error(f"Krum配置验证失败: {e}")
        return False