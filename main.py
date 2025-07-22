"""
联邦学习投毒攻击防御训练框架主程序
Federated Learning Poisoning Attack Defense Training Framework
"""

import sys
import os
import time
import threading
import webbrowser
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import ConfigManager
from utils.logger import Logger
from data.data_loader import DataManager
from data.data_splitter import DataSplitter
from models.model_manager import ModelManager
from federated.server import FederatedServer
from federated.client import FederatedClient
from federated.client_manager import ClientManager
from attacks.data_poison import DataPoisonAttack
from attacks.model_poison import ModelPoisonAttack
from web.app import WebApp


class FederatedLearningFramework:
    """联邦学习框架主类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化联邦学习框架
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_manager = None
        self.logger = None
        self.data_manager = None
        self.data_splitter = None
        self.model_manager = None
        self.server = None
        self.client_manager = None
        self.web_app = None
        
        # 训练状态
        self.training_completed = False
        self.results_file = None
        
    def initialize_system(self) -> bool:
        """
        初始化系统组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            print("联邦学习投毒攻击防御训练框架")
            print("Federated Learning Poisoning Attack Defense Training Framework")
            print("=" * 60)
            
            # 1. 初始化配置管理器
            print("正在加载配置文件...")
            self.config_manager = ConfigManager(self.config_path)
            self.config_manager.validate_config()
            print("✓ 配置文件加载成功")
            
            # 2. 初始化日志记录器
            logging_config = self.config_manager.get_logging_config()
            self.logger = Logger(
                log_dir=logging_config.get('log_dir', './logs'),
                results_dir=logging_config.get('results_dir', './results')
            )
            print("✓ 日志记录器初始化成功")
            
            # 3. 设置实验信息
            experiment_info = self._get_experiment_info()
            self.logger.set_experiment_info(experiment_info)
            
            # 4. 记录实验配置
            self._log_experiment_config(experiment_info)
            
            print("✓ 系统初始化完成")
            return True
            
        except Exception as e:
            print(f"✗ 系统初始化失败: {e}")
            return False
    
    def _get_experiment_info(self) -> Dict[str, Any]:
        """获取实验信息"""
        dataset_config = self.config_manager.get_dataset_config()
        model_config = self.config_manager.get_model_config()
        federated_config = self.config_manager.get_federated_config()
        attack_config = self.config_manager.get_attack_config()
        defense_config = self.config_manager.get_defense_config()
        
        return {
            "dataset": dataset_config.get('name'),
            "model": model_config.get('name'),
            "num_clients": federated_config.get('num_clients'),
            "num_rounds": federated_config.get('num_rounds'),
            "clients_per_round": federated_config.get('clients_per_round'),
            "local_epochs": federated_config.get('local_epochs'),
            "learning_rate": federated_config.get('learning_rate'),
            "aggregation": federated_config.get('aggregation'),
            "attack_enabled": attack_config.get('enable', False),
            "attack_type": attack_config.get('attack_type'),
            "malicious_clients": attack_config.get('malicious_clients', []),
            "defense_enabled": defense_config.get('enable', False),
            "defense_strategy": defense_config.get('strategy'),
            "data_distribution": dataset_config.get('distribution', 'iid')
        }
    
    def _log_experiment_config(self, experiment_info: Dict[str, Any]) -> None:
        """记录实验配置"""
        self.logger.log_info("=" * 50)
        self.logger.log_info("实验配置信息")
        self.logger.log_info("=" * 50)
        self.logger.log_info(f"数据集: {experiment_info['dataset']}")
        self.logger.log_info(f"数据分布: {experiment_info['data_distribution']}")
        self.logger.log_info(f"模型: {experiment_info['model']}")
        self.logger.log_info(f"客户端数量: {experiment_info['num_clients']}")
        self.logger.log_info(f"训练轮次: {experiment_info['num_rounds']}")
        self.logger.log_info(f"每轮参与客户端: {experiment_info['clients_per_round']}")
        self.logger.log_info(f"本地训练轮数: {experiment_info['local_epochs']}")
        self.logger.log_info(f"学习率: {experiment_info['learning_rate']}")
        self.logger.log_info(f"聚合方法: {experiment_info['aggregation']}")
        
        if experiment_info['attack_enabled']:
            self.logger.log_info(f"攻击类型: {experiment_info['attack_type']}")
            self.logger.log_info(f"恶意客户端: {experiment_info['malicious_clients']}")
        else:
            self.logger.log_info("攻击: 未启用")
        
        if experiment_info['defense_enabled']:
            self.logger.log_info(f"防御策略: {experiment_info['defense_strategy']}")
        else:
            self.logger.log_info("防御: 未启用")
        
        self.logger.log_info("=" * 50)
        
        # 同时在控制台输出关键信息
        print(f"\n实验配置:")
        print(f"  数据集: {experiment_info['dataset']} ({experiment_info['data_distribution']})")
        print(f"  模型: {experiment_info['model']}")
        print(f"  客户端: {experiment_info['num_clients']} (每轮 {experiment_info['clients_per_round']})")
        print(f"  训练: {experiment_info['num_rounds']} 轮 x {experiment_info['local_epochs']} 本地轮数")
        print(f"  聚合: {experiment_info['aggregation']}")
        
        if experiment_info['attack_enabled']:
            print(f"  攻击: {experiment_info['attack_type']} (客户端 {experiment_info['malicious_clients']})")
        
        if experiment_info['defense_enabled']:
            print(f"  防御: {experiment_info['defense_strategy']}")
    
    def prepare_data(self) -> bool:
        """
        准备数据集
        
        Returns:
            bool: 数据准备是否成功
        """
        try:
            print("\n正在准备数据集...")
            
            # 1. 初始化数据管理器
            dataset_config = self.config_manager.get_dataset_config()
            self.data_manager = DataManager(dataset_config)
            
            # 2. 加载数据集
            train_data, train_labels, test_data, test_labels = self.data_manager.load_dataset()
            self.logger.log_info(f"数据集加载完成 - 训练集: {len(train_data)}, 测试集: {len(test_data)}")
            
            # 3. 初始化数据分割器
            federated_config = self.config_manager.get_federated_config()
            self.data_splitter = DataSplitter(save_dir="./client_data")
            
            # 4. 分割数据
            num_clients = federated_config.get('num_clients')
            distribution = dataset_config.get('distribution', 'iid')
            
            if distribution.lower() == 'iid':
                train_client_data = self.data_splitter.split_data_iid(train_data, train_labels, num_clients)
                test_client_data = self.data_splitter.split_data_iid(test_data, test_labels, num_clients)
            else:  # non_iid
                alpha = dataset_config.get('alpha', 0.5)
                train_client_data = self.data_splitter.split_data_non_iid(train_data, train_labels, num_clients, alpha)
                test_client_data = self.data_splitter.split_data_non_iid(test_data, test_labels, num_clients, alpha)
            
            # 组合训练和测试数据
            client_data = []
            for i in range(num_clients):
                client_data.append((
                    train_client_data[i]['data'],
                    train_client_data[i]['labels'],
                    test_client_data[i]['data'],
                    test_client_data[i]['labels']
                ))
            
            self.client_data = client_data
            
            self.logger.log_info(f"数据分割完成 - {len(client_data)} 个客户端")
            print("✓ 数据集准备完成")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"数据准备失败: {e}")
            print(f"✗ 数据准备失败: {e}")
            return False
    
    def setup_models(self) -> bool:
        """
        设置模型
        
        Returns:
            bool: 模型设置是否成功
        """
        try:
            print("正在设置模型...")
            
            # 1. 初始化模型管理器
            model_config = self.config_manager.get_model_config()
            self.model_manager = ModelManager(model_config)
            
            # 2. 获取数据集信息
            dataset_info = self.data_manager.get_dataset_info()
            
            # 3. 创建全局模型
            global_model = self.model_manager.create_model(
                model_name=model_config.get('name'),
                input_shape=dataset_info['input_shape'],
                num_classes=dataset_info['num_classes']
            )
            
            self.global_model = global_model
            
            # 4. 记录模型信息
            model_summary = self.model_manager.model_summary(global_model, dataset_info['input_shape'])
            self.logger.log_info(f"模型创建完成:\n{model_summary}")
            
            print("✓ 模型设置完成")
            return True
            
        except Exception as e:
            self.logger.log_error(f"模型设置失败: {e}")
            print(f"✗ 模型设置失败: {e}")
            return False
    
    def setup_federated_components(self) -> bool:
        """
        设置联邦学习组件
        
        Returns:
            bool: 组件设置是否成功
        """
        try:
            print("正在设置联邦学习组件...")
            
            # 1. 创建服务器
            federated_config = self.config_manager.get_federated_config()
            defense_config = self.config_manager.get_defense_config()
            
            aggregation_method = federated_config.get('aggregation', 'fedavg')
            if defense_config.get('enable', False):
                aggregation_method = defense_config.get('strategy', 'fedavg')
            
            self.server = FederatedServer(
                global_model=self.global_model,
                aggregation_method=aggregation_method,
                config=defense_config
            )
            
            # 2. 创建客户端
            clients = []
            attack_config = self.config_manager.get_attack_config()
            malicious_clients = attack_config.get('malicious_clients', []) if attack_config.get('enable', False) else []
            
            for client_id, (train_data, train_labels, test_data, test_labels) in enumerate(self.client_data):
                # 创建客户端模型
                client_model = self.model_manager.clone_model(self.global_model)
                
                # 创建客户端
                client = FederatedClient(
                    client_id=client_id,
                    train_data=train_data,
                    train_labels=train_labels,
                    test_data=test_data,
                    test_labels=test_labels,
                    model=client_model
                )
                
                clients.append(client)
            
            # 3. 创建客户端管理器
            self.client_manager = ClientManager(
                clients=clients,
                selection_strategy="random"
            )
            
            self.logger.log_info(f"联邦学习组件设置完成 - 服务器: {aggregation_method}, 客户端: {len(clients)}")
            if malicious_clients:
                self.logger.log_info(f"恶意客户端: {malicious_clients}")
            
            print("✓ 联邦学习组件设置完成")
            return True
            
        except Exception as e:
            self.logger.log_error(f"联邦学习组件设置失败: {e}")
            print(f"✗ 联邦学习组件设置失败: {e}")
            return False
    
    def apply_attacks(self) -> None:
        """应用攻击策略"""
        attack_config = self.config_manager.get_attack_config()
        
        if not attack_config.get('enable', False):
            return
        
        try:
            attack_type = attack_config.get('attack_type')
            attack_params = attack_config.get('attack_params', {})
            malicious_clients = attack_config.get('malicious_clients', [])
            
            self.client_manager.apply_attacks_to_clients(
                malicious_clients=malicious_clients,
                attack_type=attack_type,
                attack_params=attack_params
            )
            
            self.logger.log_info(f"攻击策略应用完成 - 类型: {attack_type}, 客户端: {malicious_clients}")
            
        except Exception as e:
            self.logger.log_error(f"攻击策略应用失败: {e}")
    
    def run_federated_training(self) -> bool:
        """
        运行联邦学习训练
        
        Returns:
            bool: 训练是否成功完成
        """
        try:
            print("\n开始联邦学习训练...")
            
            federated_config = self.config_manager.get_federated_config()
            num_rounds = federated_config.get('num_rounds')
            clients_per_round = federated_config.get('clients_per_round')
            local_epochs = federated_config.get('local_epochs')
            learning_rate = federated_config.get('learning_rate')
            
            # 应用攻击策略
            self.apply_attacks()
            
            # 训练循环
            for round_num in range(1, num_rounds + 1):
                print(f"\n轮次 {round_num}/{num_rounds}")
                self.logger.log_info(f"开始第 {round_num} 轮训练")
                
                # 1. 选择客户端
                selected_clients = self.client_manager.select_clients(clients_per_round)
                selected_ids = [client.client_id for client in selected_clients]
                self.logger.log_info(f"选择客户端: {selected_ids}")
                
                # 2. 广播全局模型
                global_params = self.server.broadcast_model()
                self.client_manager.broadcast_model_to_clients(selected_clients, global_params)
                
                # 3. 客户端训练
                training_results = self.client_manager.train_clients(
                    selected_clients=selected_clients,
                    epochs=local_epochs,
                    learning_rate=learning_rate,
                    round_num=round_num
                )
                
                # 4. 收集客户端模型
                client_models = self.client_manager.get_client_models(selected_clients)
                
                # 5. 服务器聚合
                aggregated_params = self.server.aggregate_models(
                    client_models=client_models,
                    client_info=training_results
                )
                
                # 6. 更新全局模型
                self.server.update_global_model(aggregated_params)
                
                # 7. 评估全局模型
                # 使用测试数据评估
                test_data = self.client_data[0][2]  # 使用第一个客户端的测试数据作为全局测试集
                test_labels = self.client_data[0][3]
                global_loss, global_accuracy = self.server.evaluate_global_model(test_data, test_labels)
                
                # 8. 评估客户端
                client_eval_results = self.client_manager.evaluate_clients(selected_clients)
                
                # 9. 记录结果
                self.logger.log_round_results(
                    round_num=round_num,
                    global_loss=global_loss,
                    global_acc=global_accuracy,
                    client_results=training_results,
                    client_eval_results=client_eval_results
                )
                
                # 10. 输出进度
                avg_client_loss = sum(r['loss'] for r in training_results) / len(training_results)
                avg_client_acc = sum(r['accuracy'] for r in training_results) / len(training_results)
                
                print(f"  全局模型 - 损失: {global_loss:.4f}, 准确率: {global_accuracy:.4f}")
                print(f"  客户端平均 - 损失: {avg_client_loss:.4f}, 准确率: {avg_client_acc:.4f}")
                
                # 11. 进入下一轮
                self.server.next_round()
            
            # 保存最终结果
            self.results_file = self.logger.save_final_results()
            self.training_completed = True
            
            print(f"\n✓ 联邦学习训练完成！结果已保存到: {self.results_file}")
            self.logger.log_info("联邦学习训练完成")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"联邦学习训练失败: {e}")
            print(f"✗ 联邦学习训练失败: {e}")
            return False
    
    def start_web_interface(self) -> None:
        """启动Web界面"""
        try:
            web_config = self.config_manager.get_web_config()
            
            if not web_config.get('enable', True):
                print("Web界面已禁用")
                return
            
            print("\n正在启动Web可视化界面...")
            
            # 创建Web应用
            results_dir = self.logger.results_dir if self.logger else "./results"
            self.web_app = WebApp(results_dir=results_dir)
            
            host = web_config.get('host', 'localhost')
            port = web_config.get('port', 5000)
            
            # 检查端口可用性并自动调整
            port = self._find_available_port(host, port)
            
            # 在新线程中启动Web服务器
            def run_web_server():
                try:
                    self.web_app.run_server(host=host, port=port, debug=False)
                except Exception as e:
                    print(f"Web服务器启动失败: {e}")
                    if self.logger:
                        self.logger.log_error(f"Web服务器启动失败: {e}")
            
            web_thread = threading.Thread(target=run_web_server, daemon=True)
            web_thread.start()
            
            # 等待服务器启动
            time.sleep(3)
            
            # 验证服务器是否成功启动
            if self._check_web_server_health(host, port):
                # 自动打开浏览器
                try:
                    url = f"http://{host}:{port}"
                    webbrowser.open(url)
                    print(f"✓ Web界面已启动: {url}")
                    if self.logger:
                        self.logger.log_info(f"Web界面已启动: {url}")
                except Exception as e:
                    print(f"无法自动打开浏览器: {e}")
                    print(f"请手动访问: http://{host}:{port}")
            else:
                print("Web服务器启动可能失败，请检查日志")
            
        except Exception as e:
            print(f"Web界面启动失败: {e}")
            if self.logger:
                self.logger.log_error(f"Web界面启动失败: {e}")
    
    def _find_available_port(self, host: str, start_port: int) -> int:
        """查找可用端口"""
        import socket
        
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind((host, port))
                    return port
            except OSError:
                continue
        
        # 如果没有找到可用端口，返回原始端口
        print(f"警告: 无法找到可用端口，使用默认端口 {start_port}")
        return start_port
    
    def _check_web_server_health(self, host: str, port: int) -> bool:
        """检查Web服务器健康状态"""
        import socket
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False
    
    def save_checkpoint(self, round_num: int) -> str:
        """
        保存训练检查点
        
        Args:
            round_num: 当前轮次
            
        Returns:
            str: 检查点文件路径
        """
        try:
            checkpoint_dir = os.path.join(self.logger.results_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_round_{round_num}.pt")
            
            checkpoint_data = {
                'round_num': round_num,
                'global_model_state': self.server.get_global_model_state_dict(),
                'server_stats': self.server.get_server_statistics(),
                'client_stats': self.client_manager.get_client_statistics(),
                'experiment_info': self._get_experiment_info()
            }
            
            import torch
            torch.save(checkpoint_data, checkpoint_path)
            
            self.logger.log_info(f"检查点已保存: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.log_error(f"保存检查点失败: {e}")
            return ""
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载训练检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(checkpoint_path):
                return False
            
            import torch
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # 恢复服务器状态
            self.server.set_global_model_parameters(checkpoint_data['global_model_state'])
            self.server.current_round = checkpoint_data['round_num']
            
            self.logger.log_info(f"检查点已加载: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.log_error(f"加载检查点失败: {e}")
            return False
    
    def handle_training_interruption(self, round_num: int) -> None:
        """
        处理训练中断
        
        Args:
            round_num: 中断时的轮次
        """
        try:
            print(f"\n训练在第 {round_num} 轮被中断")
            
            # 保存当前状态
            checkpoint_path = self.save_checkpoint(round_num)
            if checkpoint_path:
                print(f"当前状态已保存到: {checkpoint_path}")
            
            # 保存部分结果
            if self.logger:
                partial_results = self.logger.save_partial_results(round_num)
                if partial_results:
                    print(f"部分结果已保存到: {partial_results}")
            
            self.logger.log_info(f"训练中断处理完成，轮次: {round_num}")
            
        except Exception as e:
            print(f"处理训练中断失败: {e}")
    
    def auto_recovery_training(self) -> bool:
        """
        自动恢复训练
        
        Returns:
            bool: 恢复是否成功
        """
        try:
            checkpoint_dir = os.path.join(self.logger.results_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                return False
            
            # 查找最新的检查点
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_round_")]
            if not checkpoint_files:
                return False
            
            # 按轮次排序，获取最新的
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
            
            print(f"发现检查点文件: {latest_checkpoint}")
            
            # 询问用户是否恢复
            try:
                response = input("是否从检查点恢复训练? (y/n): ").lower().strip()
                if response == 'y' or response == 'yes':
                    return self.load_checkpoint(latest_checkpoint)
            except (EOFError, KeyboardInterrupt):
                return False
            
            return False
            
        except Exception as e:
            print(f"自动恢复失败: {e}")
            return False
    
    def monitor_system_resources(self) -> Dict[str, Any]:
        """
        监控系统资源使用情况
        
        Returns:
            dict: 资源使用信息
        """
        try:
            import psutil
            import torch
            
            # CPU和内存信息
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU信息（如果可用）
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    'gpu_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'memory_allocated': torch.cuda.memory_allocated(),
                    'memory_reserved': torch.cuda.memory_reserved()
                }
            
            resource_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'gpu_info': gpu_info,
                'timestamp': time.time()
            }
            
            return resource_info
            
        except ImportError:
            # psutil不可用时返回基本信息
            return {'status': 'monitoring_unavailable'}
        except Exception as e:
            return {'error': str(e)}
    
    def cleanup_resources(self) -> None:
        """清理资源"""
        try:
            # 清理GPU内存
            if hasattr(self, 'global_model') and self.global_model is not None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            # 清理临时文件
            temp_dirs = ['./temp', './cache']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
            
            if self.logger:
                self.logger.log_info("系统资源清理完成")
            print("✓ 资源清理完成")
            
        except Exception as e:
            print(f"资源清理失败: {e}")
    
    def validate_system_requirements(self) -> bool:
        """
        验证系统要求
        
        Returns:
            bool: 系统要求是否满足
        """
        try:
            print("正在验证系统要求...")
            
            # 检查Python版本
            import sys
            if sys.version_info < (3, 7):
                print("✗ Python版本过低，需要3.7或更高版本")
                return False
            
            # 检查必需的包
            required_packages = [
                'torch', 'torchvision', 'numpy', 'matplotlib', 
                'flask', 'yaml', 'scipy'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"✗ 缺少必需的包: {', '.join(missing_packages)}")
                print("请运行: pip install -r requirements.txt")
                return False
            
            # 检查磁盘空间
            import shutil
            free_space = shutil.disk_usage('.').free
            required_space = 1024 * 1024 * 1024  # 1GB
            
            if free_space < required_space:
                print(f"✗ 磁盘空间不足，需要至少1GB空间")
                return False
            
            print("✓ 系统要求验证通过")
            return True
            
        except Exception as e:
            print(f"系统要求验证失败: {e}")
            return False
    
    def run(self) -> bool:
        """
        运行完整的联邦学习流程
        
        Returns:
            bool: 运行是否成功
        """
        try:
            # 1. 初始化系统
            if not self.initialize_system():
                return False
            
            # 2. 准备数据
            if not self.prepare_data():
                return False
            
            # 3. 设置模型
            if not self.setup_models():
                return False
            
            # 4. 设置联邦学习组件
            if not self.setup_federated_components():
                return False
            
            # 5. 运行训练
            if not self.run_federated_training():
                return False
            
            # 6. 启动Web界面
            self.start_web_interface()
            
            return True
            
        except KeyboardInterrupt:
            print("\n用户中断训练")
            self.logger.log_info("用户中断训练") if self.logger else None
            return False
        except Exception as e:
            print(f"运行失败: {e}")
            if self.logger:
                self.logger.log_error(f"运行失败: {e}")
            return False
        finally:
            self.cleanup_resources()


def main():
    """主程序入口"""
    try:
        # 创建联邦学习框架实例
        framework = FederatedLearningFramework()
        
        # 运行完整流程
        success = framework.run()
        
        if success:
            print("\n" + "=" * 60)
            print("联邦学习训练流程完成！")
            print("=" * 60)
            
            if framework.training_completed:
                print("训练已完成，Web界面正在运行中...")
                print("按 Ctrl+C 退出程序")
                
                # 保持程序运行以维持Web服务器
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n程序退出")
            
            sys.exit(0)
        else:
            print("\n联邦学习训练流程失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"程序运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()