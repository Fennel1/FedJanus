"""
数据分割器模块
支持IID和Non-IID数据分割，包括迪利克雷分布
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict


class DataSplitter:
    """数据分割器类，支持IID和Non-IID数据分割"""
    
    def __init__(self, save_dir: str = "./client_data"):
        """
        初始化数据分割器
        
        Args:
            save_dir: 客户端数据保存目录
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def split_data_iid(self, data: np.ndarray, labels: np.ndarray, num_clients: int) -> List[Dict[str, np.ndarray]]:
        """
        IID数据分割 - 每个客户端获得相同分布的数据
        
        Args:
            data: 训练数据
            labels: 训练标签
            num_clients: 客户端数量
            
        Returns:
            list: 每个客户端的数据字典列表
        """
        print(f"执行IID数据分割，客户端数量: {num_clients}")
        
        # 获取数据总数
        num_samples = len(data)
        samples_per_client = num_samples // num_clients
        
        # 随机打乱数据索引
        indices = np.random.permutation(num_samples)
        
        client_data_list = []
        
        for i in range(num_clients):
            # 计算每个客户端的数据索引范围
            start_idx = i * samples_per_client
            if i == num_clients - 1:  # 最后一个客户端获得剩余所有数据
                end_idx = num_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            # 获取客户端数据索引
            client_indices = indices[start_idx:end_idx]
            
            # 提取客户端数据
            client_data = {
                'data': data[client_indices],
                'labels': labels[client_indices],
                'client_id': i,
                'num_samples': len(client_indices)
            }
            
            client_data_list.append(client_data)
            
            print(f"客户端 {i}: {len(client_indices)} 个样本")
        
        return client_data_list
    
    def split_data_non_iid(self, data: np.ndarray, labels: np.ndarray, 
                          num_clients: int, alpha: float = 0.5) -> List[Dict[str, np.ndarray]]:
        """
        Non-IID数据分割 - 使用迪利克雷分布创建不平衡分布
        
        Args:
            data: 训练数据
            labels: 训练标签
            num_clients: 客户端数量
            alpha: 迪利克雷分布参数，越小越不平衡
            
        Returns:
            list: 每个客户端的数据字典列表
        """
        print(f"执行Non-IID数据分割，客户端数量: {num_clients}, alpha: {alpha}")
        
        # 获取类别信息
        num_classes = len(np.unique(labels))
        print(f"数据集类别数: {num_classes}")
        
        # 按类别组织数据索引
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # 为每个类别生成迪利克雷分布
        client_data_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            # 获取当前类别的所有样本索引
            indices = np.array(class_indices[class_id])
            num_samples_class = len(indices)
            
            # 生成迪利克雷分布
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # 根据比例分配样本
            proportions = np.cumsum(proportions)
            
            # 随机打乱当前类别的样本
            np.random.shuffle(indices)
            
            start_idx = 0
            for client_id in range(num_clients):
                # 计算当前客户端应该获得的样本数量
                end_proportion = proportions[client_id]
                end_idx = int(end_proportion * num_samples_class)
                
                # 确保不超出范围
                end_idx = min(end_idx, num_samples_class)
                
                # 分配样本给客户端
                if start_idx < end_idx:
                    client_data_indices[client_id].extend(indices[start_idx:end_idx])
                
                start_idx = end_idx
        
        # 构建客户端数据
        client_data_list = []
        
        for client_id in range(num_clients):
            indices = client_data_indices[client_id]
            
            if len(indices) == 0:
                print(f"警告: 客户端 {client_id} 没有分配到数据")
                # 为空客户端分配少量随机数据
                indices = np.random.choice(len(data), size=10, replace=False)
            
            indices = np.array(indices)
            
            # 提取客户端数据
            client_data = {
                'data': data[indices],
                'labels': labels[indices],
                'client_id': client_id,
                'num_samples': len(indices)
            }
            
            client_data_list.append(client_data)
            
            # 统计客户端的类别分布
            unique_labels, counts = np.unique(labels[indices], return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"客户端 {client_id}: {len(indices)} 个样本, 类别分布: {label_dist}")
        
        return client_data_list
    
    def save_client_data(self, client_data_list: List[Dict[str, np.ndarray]], 
                        experiment_name: str = "default", 
                        dataset_info: Dict[str, Any] = None,
                        split_config: Dict[str, Any] = None) -> str:
        """
        保存客户端数据到本地文件
        
        Args:
            client_data_list: 客户端数据列表
            experiment_name: 实验名称
            dataset_info: 数据集信息
            split_config: 分割配置信息
            
        Returns:
            str: 保存路径
        """
        import json
        from datetime import datetime
        
        save_path = os.path.join(self.save_dir, experiment_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存每个客户端的数据
        for client_data in client_data_list:
            client_id = client_data['client_id']
            filename = f"client_{client_id}.pkl"
            filepath = os.path.join(save_path, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(client_data, f)
        
        # 获取详细统计信息
        stats = self.get_split_statistics(client_data_list)
        
        # 创建详细的元数据
        metadata = {
            'experiment_info': {
                'name': experiment_name,
                'created_time': datetime.now().isoformat(),
                'total_clients': len(client_data_list),
                'total_samples': stats['total_samples']
            },
            'dataset_info': dataset_info or {},
            'split_config': split_config or {},
            'statistics': {
                'samples_per_client': stats['samples_per_client'],
                'avg_samples_per_client': float(stats['avg_samples_per_client']),
                'std_samples_per_client': float(stats['std_samples_per_client']),
                'min_samples_per_client': int(stats['min_samples_per_client']),
                'max_samples_per_client': int(stats['max_samples_per_client']),
                'global_label_distribution': {str(k): int(v) for k, v in stats['global_label_distribution'].items()}
            },
            'client_details': []
        }
        
        # 添加每个客户端的详细信息
        for i, client_data in enumerate(client_data_list):
            labels = client_data['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}
            
            # 计算数据分布的不平衡程度（基尼系数）
            proportions = counts / len(labels)
            gini_coefficient = 1 - np.sum(proportions ** 2)
            
            client_info = {
                'client_id': int(client_data['client_id']),
                'num_samples': int(client_data['num_samples']),
                'data_shape': list(client_data['data'].shape),
                'label_distribution': label_dist,
                'num_classes': len(unique_labels),
                'gini_coefficient': float(gini_coefficient),
                'data_statistics': {
                    'mean': float(client_data['data'].mean()),
                    'std': float(client_data['data'].std()),
                    'min': float(client_data['data'].min()),
                    'max': float(client_data['data'].max())
                }
            }
            
            metadata['client_details'].append(client_info)
        
        # 保存JSON格式的元数据
        meta_json_path = os.path.join(save_path, 'meta.json')
        with open(meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 保存pickle格式的元数据（向后兼容）
        simple_metadata = {
            'num_clients': len(client_data_list),
            'total_samples': stats['total_samples'],
            'experiment_name': experiment_name
        }
        
        metadata_path = os.path.join(save_path, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(simple_metadata, f)
        
        print(f"客户端数据已保存到: {save_path}")
        print(f"元数据文件: meta.json, metadata.pkl")
        return save_path
    
    def load_client_data(self, experiment_name: str = "default") -> List[Dict[str, np.ndarray]]:
        """
        从本地文件加载客户端数据
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            list: 客户端数据列表
        """
        load_path = os.path.join(self.save_dir, experiment_name)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到实验数据: {load_path}")
        
        # 加载元数据
        metadata_path = os.path.join(load_path, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        num_clients = metadata['num_clients']
        
        # 加载客户端数据
        client_data_list = []
        for client_id in range(num_clients):
            filename = f"client_{client_id}.pkl"
            filepath = os.path.join(load_path, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"找不到客户端数据文件: {filepath}")
            
            with open(filepath, 'rb') as f:
                client_data = pickle.load(f)
            
            client_data_list.append(client_data)
        
        print(f"已加载 {num_clients} 个客户端的数据")
        return client_data_list
    
    def get_split_statistics(self, client_data_list: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        获取数据分割统计信息
        
        Args:
            client_data_list: 客户端数据列表
            
        Returns:
            dict: 统计信息
        """
        num_clients = len(client_data_list)
        total_samples = sum(cd['num_samples'] for cd in client_data_list)
        
        # 计算每个客户端的样本数量
        samples_per_client = [cd['num_samples'] for cd in client_data_list]
        
        # 计算类别分布统计
        all_labels = []
        client_label_distributions = []
        
        for client_data in client_data_list:
            labels = client_data['labels']
            all_labels.extend(labels)
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique_labels.astype(int), counts.astype(int)))
            client_label_distributions.append(label_dist)
        
        # 全局类别分布
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        global_label_dist = dict(zip(unique_labels.astype(int), counts.astype(int)))
        
        statistics = {
            'num_clients': num_clients,
            'total_samples': total_samples,
            'samples_per_client': samples_per_client,
            'avg_samples_per_client': np.mean(samples_per_client),
            'std_samples_per_client': np.std(samples_per_client),
            'min_samples_per_client': np.min(samples_per_client),
            'max_samples_per_client': np.max(samples_per_client),
            'global_label_distribution': global_label_dist,
            'client_label_distributions': client_label_distributions
        }
        
        return statistics
    
    def list_saved_experiments(self) -> List[str]:
        """
        列出所有已保存的实验
        
        Returns:
            list: 实验名称列表
        """
        if not os.path.exists(self.save_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.save_dir):
            item_path = os.path.join(self.save_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.pkl')
                if os.path.exists(metadata_path):
                    experiments.append(item)
        
        return experiments


class FederatedDataManager:
    """联邦数据管理器，整合数据加载和分割功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化联邦数据管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.splitter = DataSplitter()
    
    def prepare_federated_data(self, train_data: np.ndarray, train_labels: np.ndarray,
                             test_data: np.ndarray, test_labels: np.ndarray,
                             save_data: bool = False, experiment_name: str = None) -> Tuple[List[Dict], Dict]:
        """
        准备联邦学习数据
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签
            test_data: 测试数据
            test_labels: 测试标签
            save_data: 是否保存数据到本地
            experiment_name: 实验名称（如果保存数据）
            
        Returns:
            tuple: (客户端训练数据列表, 全局测试数据)
        """
        num_clients = self.config.get('num_clients', 10)
        distribution = self.config.get('distribution', 'iid')
        alpha = self.config.get('alpha', 0.5)
        
        # 分割训练数据
        if distribution.lower() == 'iid':
            client_data_list = self.splitter.split_data_iid(train_data, train_labels, num_clients)
        elif distribution.lower() == 'non_iid':
            client_data_list = self.splitter.split_data_non_iid(train_data, train_labels, num_clients, alpha)
        else:
            raise ValueError(f"不支持的数据分布类型: {distribution}")
        
        # 全局测试数据
        global_test_data = {
            'data': test_data,
            'labels': test_labels
        }
        
        # 如果需要保存数据
        if save_data:
            if experiment_name is None:
                experiment_name = f"{self.config.get('name', 'dataset')}_{distribution}_clients_{num_clients}"
            
            # 准备数据集信息
            dataset_info = {
                'name': self.config.get('name', 'unknown'),
                'num_classes': len(np.unique(train_labels)),
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'data_shape': list(train_data.shape[1:]),
                'input_shape': self.config.get('input_shape', list(train_data.shape[1:])),
                'data_dir': self.config.get('data_dir', './datasets')
            }
            
            # 准备分割配置信息
            split_config = {
                'distribution_type': distribution,
                'num_clients': num_clients,
                'alpha': alpha if distribution.lower() == 'non_iid' else None,
                'random_seed': None  # 可以在未来添加随机种子支持
            }
            
            # 保存客户端数据
            self.splitter.save_client_data(
                client_data_list, 
                experiment_name, 
                dataset_info, 
                split_config
            )
        
        return client_data_list, global_test_data