"""
配置管理器
Configuration Manager for loading and managing YAML configuration files
"""

import yaml
import os
from typing import Dict, Any


class ConfigManager:
    """配置管理器类，负责加载和管理YAML配置文件"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                if config is None:
                    raise ValueError("配置文件为空")
                return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML配置文件格式错误: {e}")
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """
        获取数据集配置
        
        Returns:
            数据集配置字典
        """
        return self.config.get('dataset', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        return self.config.get('model', {})
    
    def get_federated_config(self) -> Dict[str, Any]:
        """
        获取联邦学习配置
        
        Returns:
            联邦学习配置字典
        """
        return self.config.get('federated', {})
    
    def get_attack_config(self) -> Dict[str, Any]:
        """
        获取攻击配置
        
        Returns:
            攻击配置字典
        """
        return self.config.get('attack', {})
    
    def get_defense_config(self) -> Dict[str, Any]:
        """
        获取防御配置
        
        Returns:
            防御配置字典
        """
        return self.config.get('defense', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            日志配置字典
        """
        return self.config.get('logging', {})
    
    def get_web_config(self) -> Dict[str, Any]:
        """
        获取Web配置
        
        Returns:
            Web配置字典
        """
        return self.config.get('web', {})
    
    def get_config(self, key: str = None) -> Any:
        """
        获取指定配置项或全部配置
        
        Args:
            key: 配置项键名，如果为None则返回全部配置
            
        Returns:
            配置值或全部配置字典
        """
        if key is None:
            return self.config
        return self.config.get(key)
    
    def validate_config(self) -> bool:
        """
        验证配置文件的必需参数
        
        Returns:
            配置是否有效
        """
        required_sections = ['dataset', 'model', 'federated', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"缺少必需的配置节: {section}")
        
        # 验证数据集配置
        dataset_config = self.get_dataset_config()
        if 'name' not in dataset_config:
            raise ValueError("数据集配置缺少name参数")
        
        # 验证模型配置
        model_config = self.get_model_config()
        if 'name' not in model_config:
            raise ValueError("模型配置缺少name参数")
        
        # 验证联邦学习配置
        federated_config = self.get_federated_config()
        required_fed_params = ['num_clients', 'num_rounds', 'local_epochs', 'learning_rate']
        for param in required_fed_params:
            if param not in federated_config:
                raise ValueError(f"联邦学习配置缺少{param}参数")
        
        return True