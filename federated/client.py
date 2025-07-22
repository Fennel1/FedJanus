"""
联邦学习客户端实现
Client implementation for federated learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import copy
import logging


class FederatedClient:
    """
    联邦学习客户端类
    负责本地训练、模型评估和参数交换
    """
    
    def __init__(self, 
                 client_id: int,
                 train_data: np.ndarray,
                 train_labels: np.ndarray,
                 test_data: np.ndarray,
                 test_labels: np.ndarray,
                 model: nn.Module,
                 device: torch.device = None):
        """
        初始化联邦学习客户端
        
        Args:
            client_id: 客户端ID
            train_data: 训练数据
            train_labels: 训练标签
            test_data: 测试数据
            test_labels: 测试标签
            model: 神经网络模型
            device: 计算设备
        """
        self.client_id = client_id
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据处理
        self.train_data = torch.FloatTensor(train_data).to(self.device)
        self.train_labels = torch.LongTensor(train_labels).to(self.device)
        self.test_data = torch.FloatTensor(test_data).to(self.device)
        self.test_labels = torch.LongTensor(test_labels).to(self.device)
        
        # 创建数据集和数据加载器
        self.train_dataset = TensorDataset(self.train_data, self.train_labels)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        
        # 模型设置
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史记录
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'rounds': []
        }
        
        # 攻击相关设置
        self.is_malicious = False
        self.attack_config = None
        
        # 日志设置
        self.logger = logging.getLogger(f'Client_{client_id}')
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        设置模型参数（从服务器接收）
        
        Args:
            parameters: 模型参数字典
        """
        model_dict = self.model.state_dict()
        
        # 验证参数键是否匹配
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name].copy_(param.to(self.device))
            else:
                self.logger.warning(f"参数键 {name} 在模型中不存在")
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取模型参数（发送给服务器）
        
        Returns:
            模型参数字典
        """
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def get_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取完整的模型状态字典
        
        Returns:
            模型状态字典
        """
        return {name: param.clone().detach() for name, param in self.model.state_dict().items()}
    
    def train(self, 
              epochs: int = 5, 
              learning_rate: float = 0.01, 
              batch_size: int = 32,
              round_num: int = 0) -> Dict[str, Any]:
        """
        执行本地训练
        
        Args:
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            round_num: 联邦学习轮次
            
        Returns:
            训练结果字典
        """
        self.model.train()
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=False
        )
        
        # 设置优化器
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 统计信息
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 记录每个epoch的结果
            epoch_avg_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            
            self.logger.debug(f"客户端 {self.client_id}, 轮次 {round_num}, Epoch {epoch+1}/{epochs}: "
                            f"Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct_predictions += epoch_correct
        
        # 计算平均结果
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = correct_predictions / total_samples
        else:
            avg_loss = 0.0
            avg_accuracy = 0.0
        
        # 记录训练历史
        self.training_history['loss'].append(avg_loss)
        self.training_history['accuracy'].append(avg_accuracy)
        self.training_history['rounds'].append(round_num)
        
        training_result = {
            'client_id': self.client_id,
            'round': round_num,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'samples': len(self.train_dataset),
            'epochs': epochs,
            'learning_rate': learning_rate
        }
        
        self.logger.info(f"客户端 {self.client_id} 训练完成 - 轮次: {round_num}, "
                        f"损失: {avg_loss:.4f}, 准确率: {avg_accuracy:.4f}")
        
        return training_result
    
    def evaluate(self, use_test_data: bool = True) -> Tuple[float, float]:
        """
        评估模型性能
        
        Args:
            use_test_data: 是否使用测试数据，False则使用训练数据
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.eval()
        
        # 选择评估数据
        if use_test_data:
            eval_dataset = self.test_dataset
            data_type = "测试"
        else:
            eval_dataset = self.train_dataset
            data_type = "训练"
        
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        self.logger.debug(f"客户端 {self.client_id} {data_type}数据评估: "
                         f"损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def apply_attack(self, attack_type: str, attack_params: Dict[str, Any]) -> None:
        """
        应用攻击策略
        
        Args:
            attack_type: 攻击类型
            attack_params: 攻击参数
        """
        self.is_malicious = True
        self.attack_config = {
            'type': attack_type,
            'params': attack_params
        }
        
        self.logger.info(f"客户端 {self.client_id} 设置为恶意客户端，攻击类型: {attack_type}")
        
        # 根据攻击类型应用相应的攻击
        if attack_type == "label_flipping":
            self._apply_label_flipping_attack(attack_params)
        elif attack_type == "data_poisoning":
            self._apply_data_poisoning_attack(attack_params)
        else:
            self.logger.warning(f"未知的攻击类型: {attack_type}")
    
    def _apply_label_flipping_attack(self, attack_params: Dict[str, Any]) -> None:
        """
        应用标签反转攻击
        
        Args:
            attack_params: 攻击参数
        """
        flip_ratio = attack_params.get('flip_ratio', 0.1)
        
        # 随机选择要反转的样本
        num_samples = len(self.train_labels)
        num_flip = int(num_samples * flip_ratio)
        flip_indices = np.random.choice(num_samples, num_flip, replace=False)
        
        # 获取标签的唯一值
        unique_labels = torch.unique(self.train_labels)
        num_classes = len(unique_labels)
        
        # 反转标签
        for idx in flip_indices:
            original_label = self.train_labels[idx].item()
            # 随机选择一个不同的标签
            new_label = original_label
            while new_label == original_label:
                new_label = np.random.choice(unique_labels.cpu().numpy())
            self.train_labels[idx] = new_label
        
        # 更新数据集
        self.train_dataset = TensorDataset(self.train_data, self.train_labels)
        
        self.logger.info(f"客户端 {self.client_id} 应用标签反转攻击: "
                        f"反转了 {num_flip}/{num_samples} 个样本的标签")
    
    def _apply_data_poisoning_attack(self, attack_params: Dict[str, Any]) -> None:
        """
        应用数据投毒攻击
        
        Args:
            attack_params: 攻击参数
        """
        poison_ratio = attack_params.get('poison_ratio', 0.1)
        noise_scale = attack_params.get('noise_scale', 0.1)
        
        # 随机选择要投毒的样本
        num_samples = len(self.train_data)
        num_poison = int(num_samples * poison_ratio)
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)
        
        # 添加噪声
        for idx in poison_indices:
            noise = torch.randn_like(self.train_data[idx]) * noise_scale
            self.train_data[idx] += noise
        
        # 更新数据集
        self.train_dataset = TensorDataset(self.train_data, self.train_labels)
        
        self.logger.info(f"客户端 {self.client_id} 应用数据投毒攻击: "
                        f"投毒了 {num_poison}/{num_samples} 个样本")
    
    def get_training_history(self) -> Dict[str, List]:
        """
        获取训练历史记录
        
        Returns:
            训练历史字典
        """
        return self.training_history.copy()
    
    def reset_training_history(self) -> None:
        """
        重置训练历史记录
        """
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'rounds': []
        }
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取客户端数据统计信息
        
        Returns:
            数据统计字典
        """
        train_labels_np = self.train_labels.cpu().numpy()
        test_labels_np = self.test_labels.cpu().numpy()
        
        # 计算标签分布
        unique_train, counts_train = np.unique(train_labels_np, return_counts=True)
        unique_test, counts_test = np.unique(test_labels_np, return_counts=True)
        
        train_distribution = dict(zip(unique_train.tolist(), counts_train.tolist()))
        test_distribution = dict(zip(unique_test.tolist(), counts_test.tolist()))
        
        return {
            'client_id': self.client_id,
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'train_label_distribution': train_distribution,
            'test_label_distribution': test_distribution,
            'is_malicious': self.is_malicious,
            'attack_config': self.attack_config
        }
    
    def save_model(self, filepath: str) -> None:
        """
        保存客户端模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'client_id': self.client_id,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'is_malicious': self.is_malicious,
            'attack_config': self.attack_config
        }, filepath)
        
        self.logger.info(f"客户端 {self.client_id} 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载客户端模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.is_malicious = checkpoint.get('is_malicious', False)
        self.attack_config = checkpoint.get('attack_config', None)
        
        self.logger.info(f"客户端 {self.client_id} 模型已从 {filepath} 加载")
    
    def __str__(self) -> str:
        """
        返回客户端的字符串表示
        """
        return (f"FederatedClient(id={self.client_id}, "
                f"train_samples={len(self.train_dataset)}, "
                f"test_samples={len(self.test_dataset)}, "
                f"malicious={self.is_malicious})")
    
    def __repr__(self) -> str:
        """
        返回客户端的详细字符串表示
        """
        return self.__str__()