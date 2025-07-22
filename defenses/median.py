"""
中位数防御算法实现
Median defense algorithm implementation for federated learning
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


class MedianDefense(AggregationStrategy):
    """
    中位数防御聚合算法
    Median defense aggregation algorithm
    
    中位数防御算法通过对每个模型参数计算中位数来抵御恶意客户端的投毒攻击，
    相比于平均值，中位数对异常值更加鲁棒。
    """
    
    def __init__(self, 
                 trimmed_mean: bool = False,
                 trim_ratio: float = 0.1,
                 coordinate_wise: bool = True,
                 device: torch.device = None):
        """
        初始化中位数防御器
        
        Args:
            trimmed_mean: 是否使用修剪均值而不是中位数
            trim_ratio: 修剪比例（当使用修剪均值时）
            coordinate_wise: 是否按坐标计算中位数（True）还是按客户端计算（False）
            device: 计算设备
        """
        self.trimmed_mean = trimmed_mean
        self.trim_ratio = trim_ratio
        self.coordinate_wise = coordinate_wise
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('MedianDefense')
        
        # 验证参数
        if not isinstance(trimmed_mean, bool):
            raise ValueError("trimmed_mean 必须是布尔值")
        
        if not (0.0 <= trim_ratio <= 0.5):
            raise ValueError("trim_ratio 必须在 [0.0, 0.5] 范围内")
        
        if not isinstance(coordinate_wise, bool):
            raise ValueError("coordinate_wise 必须是布尔值")
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        执行中位数防御聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表（中位数聚合中不使用）
            **kwargs: 其他参数
            
        Returns:
            聚合后的模型参数字典
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        if num_clients < 3:
            self.logger.warning(f"客户端数量 ({num_clients}) 较少，中位数防御效果可能不佳")
        
        self.logger.debug(f"开始中位数防御聚合，客户端数量: {num_clients}")
        
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
        
        if self.coordinate_wise:
            aggregated_params = self._coordinate_wise_median(client_models)
        else:
            aggregated_params = self._client_wise_median(client_models)
        
        self.logger.debug(f"中位数防御聚合完成")
        
        return aggregated_params
    
    def _coordinate_wise_median(self, client_models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        按坐标计算中位数聚合
        
        Args:
            client_models: 客户端模型参数列表
            
        Returns:
            聚合后的模型参数字典
        """
        aggregated_params = {}
        param_keys = client_models[0].keys()
        
        for param_name in param_keys:
            # 收集所有客户端的该参数
            client_params = [model[param_name].to(self.device) for model in client_models]
            
            # 将参数堆叠成张量 (num_clients, *param_shape)
            stacked_params = torch.stack(client_params, dim=0)
            
            if self.trimmed_mean:
                # 使用修剪均值
                aggregated_param = self._compute_trimmed_mean(stacked_params)
            else:
                # 使用中位数
                aggregated_param = self._compute_median(stacked_params)
            
            aggregated_params[param_name] = aggregated_param
        
        return aggregated_params
    
    def _client_wise_median(self, client_models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        按客户端计算中位数聚合（选择中位数客户端的模型）
        
        Args:
            client_models: 客户端模型参数列表
            
        Returns:
            聚合后的模型参数字典
        """
        num_clients = len(client_models)
        
        # 计算客户端之间的距离矩阵
        distance_matrix = self._compute_distance_matrix(client_models)
        
        # 计算每个客户端到其他所有客户端的距离总和
        distance_sums = torch.sum(distance_matrix, dim=1)
        
        # 选择距离总和的中位数客户端
        median_idx = torch.argsort(distance_sums)[num_clients // 2].item()
        
        self.logger.info(f"选择客户端 {median_idx} 作为中位数客户端")
        
        # 返回中位数客户端的模型参数
        return {k: v.clone().to(self.device) for k, v in client_models[median_idx].items()}
    
    def _compute_median(self, stacked_params: torch.Tensor) -> torch.Tensor:
        """
        计算张量的中位数
        
        Args:
            stacked_params: 堆叠的参数张量 (num_clients, *param_shape)
            
        Returns:
            中位数张量
        """
        # 沿着客户端维度（dim=0）计算中位数
        median_values, _ = torch.median(stacked_params, dim=0)
        return median_values
    
    def _compute_trimmed_mean(self, stacked_params: torch.Tensor) -> torch.Tensor:
        """
        计算修剪均值
        
        Args:
            stacked_params: 堆叠的参数张量 (num_clients, *param_shape)
            
        Returns:
            修剪均值张量
        """
        num_clients = stacked_params.shape[0]
        
        # 计算需要修剪的客户端数量
        num_trim = int(num_clients * self.trim_ratio)
        
        if num_trim == 0:
            # 如果不需要修剪，直接计算均值
            return torch.mean(stacked_params, dim=0)
        
        # 对每个坐标位置进行排序
        sorted_params, _ = torch.sort(stacked_params, dim=0)
        
        # 去除最小和最大的 num_trim 个值
        if num_trim * 2 >= num_clients:
            # 如果修剪的数量太多，只保留中间的值
            start_idx = num_clients // 4
            end_idx = num_clients - num_clients // 4
        else:
            start_idx = num_trim
            end_idx = num_clients - num_trim
        
        trimmed_params = sorted_params[start_idx:end_idx]
        
        # 计算修剪后的均值
        return torch.mean(trimmed_params, dim=0)
    
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
    
    def get_name(self) -> str:
        """获取策略名称"""
        if self.trimmed_mean:
            return "trimmed_mean"
        elif self.coordinate_wise:
            return "coordinate_median"
        else:
            return "client_median"
    
    def evaluate_robustness(self, 
                           client_models: List[Dict[str, torch.Tensor]],
                           true_model: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        评估防御效果的鲁棒性
        
        Args:
            client_models: 客户端模型参数列表
            true_model: 真实模型参数（如果可用）
            
        Returns:
            鲁棒性评估指标字典
        """
        if not client_models:
            return {}
        
        num_clients = len(client_models)
        
        # 计算聚合结果
        aggregated_model = self.aggregate(client_models)
        
        # 计算距离矩阵
        distance_matrix = self._compute_distance_matrix(client_models)
        
        # 计算评估指标
        metrics = {}
        
        # 1. 客户端间距离的统计信息
        upper_triangle_distances = distance_matrix[torch.triu(torch.ones_like(distance_matrix), diagonal=1) == 1]
        metrics['mean_pairwise_distance'] = torch.mean(upper_triangle_distances).item()
        metrics['std_pairwise_distance'] = torch.std(upper_triangle_distances).item()
        metrics['median_pairwise_distance'] = torch.median(upper_triangle_distances).item()
        
        # 2. 聚合模型到各客户端的距离
        aggregated_vector = torch.cat([param.flatten() for param in aggregated_model.values()])
        client_vectors = []
        for model in client_models:
            client_vector = torch.cat([param.flatten() for param in model.values()])
            client_vectors.append(client_vector.to(self.device))
        
        distances_to_aggregated = []
        for client_vector in client_vectors:
            distance = torch.norm(aggregated_vector - client_vector, p=2).item()
            distances_to_aggregated.append(distance)
        
        metrics['mean_distance_to_aggregated'] = np.mean(distances_to_aggregated)
        metrics['std_distance_to_aggregated'] = np.std(distances_to_aggregated)
        metrics['median_distance_to_aggregated'] = np.median(distances_to_aggregated)
        
        # 3. 如果有真实模型，计算聚合模型的准确性
        if true_model is not None:
            true_vector = torch.cat([param.flatten() for param in true_model.values()])
            distance_to_true = torch.norm(aggregated_vector - true_vector.to(self.device), p=2).item()
            metrics['distance_to_true_model'] = distance_to_true
        
        # 4. 异常值检测指标
        if len(distances_to_aggregated) > 2:
            q1 = np.percentile(distances_to_aggregated, 25)
            q3 = np.percentile(distances_to_aggregated, 75)
            iqr = q3 - q1
            
            # 如果IQR太小，使用标准差方法
            if iqr < 1e-6:
                mean_dist = np.mean(distances_to_aggregated)
                std_dist = np.std(distances_to_aggregated)
                outlier_threshold = mean_dist + 2 * std_dist
            else:
                outlier_threshold = q3 + 1.5 * iqr
            
            outliers = [i for i, dist in enumerate(distances_to_aggregated) if dist > outlier_threshold]
        else:
            outliers = []
        metrics['num_outliers'] = len(outliers)
        metrics['outlier_ratio'] = len(outliers) / num_clients
        metrics['outlier_clients'] = outliers
        
        self.logger.info(f"鲁棒性评估完成，检测到 {len(outliers)} 个异常客户端")
        
        return metrics
    
    def tune_parameters(self, 
                       client_models: List[Dict[str, torch.Tensor]],
                       validation_metric_fn: Optional[callable] = None) -> Dict[str, Any]:
        """
        参数调优
        
        Args:
            client_models: 客户端模型参数列表
            validation_metric_fn: 验证指标函数，接受聚合模型并返回性能分数
            
        Returns:
            最佳参数配置字典
        """
        if not client_models:
            return {}
        
        self.logger.info("开始中位数防御参数调优")
        
        # 定义参数搜索空间
        param_configs = []
        
        # 测试不同的聚合方式
        for coordinate_wise in [True, False]:
            for trimmed_mean in [False, True]:
                if trimmed_mean:
                    # 测试不同的修剪比例
                    for trim_ratio in [0.1, 0.2, 0.3]:
                        param_configs.append({
                            'coordinate_wise': coordinate_wise,
                            'trimmed_mean': trimmed_mean,
                            'trim_ratio': trim_ratio
                        })
                else:
                    param_configs.append({
                        'coordinate_wise': coordinate_wise,
                        'trimmed_mean': trimmed_mean,
                        'trim_ratio': 0.0
                    })
        
        best_config = None
        best_score = float('-inf')
        results = []
        
        for config in param_configs:
            try:
                # 创建临时防御器
                temp_defense = MedianDefense(
                    trimmed_mean=config['trimmed_mean'],
                    trim_ratio=config['trim_ratio'],
                    coordinate_wise=config['coordinate_wise'],
                    device=self.device
                )
                
                # 执行聚合
                aggregated_model = temp_defense.aggregate(client_models)
                
                # 计算评估指标
                if validation_metric_fn is not None:
                    score = validation_metric_fn(aggregated_model)
                else:
                    # 使用默认的鲁棒性指标
                    robustness_metrics = temp_defense.evaluate_robustness(client_models)
                    # 使用负的平均距离作为分数（距离越小越好）
                    score = -robustness_metrics.get('mean_distance_to_aggregated', float('inf'))
                
                results.append({
                    'config': config,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                self.logger.debug(f"配置 {config} 得分: {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"配置 {config} 测试失败: {e}")
                continue
        
        if best_config is not None:
            self.logger.info(f"最佳配置: {best_config}, 得分: {best_score:.4f}")
            
            # 更新当前实例的参数
            self.trimmed_mean = best_config['trimmed_mean']
            self.trim_ratio = best_config['trim_ratio']
            self.coordinate_wise = best_config['coordinate_wise']
        else:
            self.logger.warning("参数调优失败，保持当前配置")
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }


def create_median_defense(config: Dict[str, Any] = None, **kwargs) -> MedianDefense:
    """
    根据配置创建中位数防御实例
    
    Args:
        config: 防御配置字典
        **kwargs: 直接传递的参数（优先级高于config）
        
    Returns:
        中位数防御实例
    """
    # 首先从config获取默认参数
    trimmed_mean = False
    trim_ratio = 0.1
    coordinate_wise = True
    
    if config is not None:
        defense_params = config.get('defense_params', {})
        trimmed_mean = defense_params.get('trimmed_mean', trimmed_mean)
        trim_ratio = defense_params.get('trim_ratio', trim_ratio)
        coordinate_wise = defense_params.get('coordinate_wise', coordinate_wise)
    
    # kwargs参数优先级更高，会覆盖config中的参数
    trimmed_mean = kwargs.get('trimmed_mean', trimmed_mean)
    trim_ratio = kwargs.get('trim_ratio', trim_ratio)
    coordinate_wise = kwargs.get('coordinate_wise', coordinate_wise)
    
    # 验证参数
    if not isinstance(trimmed_mean, bool):
        raise ValueError(f"trimmed_mean 必须是布尔值，得到: {trimmed_mean}")
    
    if not isinstance(coordinate_wise, bool):
        raise ValueError(f"coordinate_wise 必须是布尔值，得到: {coordinate_wise}")
    
    if not (0.0 <= trim_ratio <= 0.5):
        raise ValueError(f"trim_ratio 必须在 [0.0, 0.5] 范围内，得到: {trim_ratio}")
    
    return MedianDefense(
        trimmed_mean=trimmed_mean,
        trim_ratio=trim_ratio,
        coordinate_wise=coordinate_wise,
        device=kwargs.get('device')
    )


def validate_median_config(config: Dict[str, Any]) -> bool:
    """
    验证中位数防御配置的有效性
    
    Args:
        config: 防御配置字典
        
    Returns:
        配置是否有效
    """
    try:
        defense_params = config.get('defense_params', {})
        
        # 检查可选参数
        if 'trimmed_mean' in defense_params:
            trimmed_mean = defense_params['trimmed_mean']
            if not isinstance(trimmed_mean, bool):
                raise ValueError(f"trimmed_mean 必须是布尔值，得到: {trimmed_mean}")
        
        if 'coordinate_wise' in defense_params:
            coordinate_wise = defense_params['coordinate_wise']
            if not isinstance(coordinate_wise, bool):
                raise ValueError(f"coordinate_wise 必须是布尔值，得到: {coordinate_wise}")
        
        if 'trim_ratio' in defense_params:
            trim_ratio = defense_params['trim_ratio']
            if not isinstance(trim_ratio, (int, float)) or not (0.0 <= trim_ratio <= 0.5):
                raise ValueError(f"trim_ratio 必须在 [0.0, 0.5] 范围内，得到: {trim_ratio}")
        
        return True
        
    except Exception as e:
        logging.getLogger('MedianDefense').error(f"中位数防御配置验证失败: {e}")
        return False