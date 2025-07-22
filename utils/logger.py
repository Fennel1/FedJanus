"""
日志记录器
Logger for recording training process and results
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any
from pathlib import Path


class Logger:
    """日志记录器类，负责记录训练过程和结果"""
    
    def __init__(self, log_dir: str = "./logs", results_dir: str = "./results"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志文件目录
            results_dir: 结果文件目录
        """
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self._setup_logger()
        
        # 初始化结果存储
        self.training_results = {
            "experiment_info": {},
            "global_results": {
                "rounds": [],
                "global_loss": [],
                "global_accuracy": []
            },
            "client_results": {}
        }
    
    def _setup_logger(self):
        """设置Python日志记录器"""
        # 创建日志文件名（包含时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = self.log_dir / f"training_{timestamp}.log"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("日志记录器初始化完成")
    
    def set_experiment_info(self, experiment_info: Dict[str, Any]):
        """
        设置实验信息
        
        Args:
            experiment_info: 实验配置信息
        """
        self.training_results["experiment_info"] = experiment_info
        self.logger.info(f"实验信息设置: {experiment_info}")
    
    def log_round_results(self, round_num: int, global_loss: float, 
                         global_acc: float, client_results: List[Dict[str, float]],
                         client_eval_results: List[Dict[str, float]] = None):
        """
        记录每轮训练结果
        
        Args:
            round_num: 训练轮次
            global_loss: 全局损失值
            global_acc: 全局准确率
            client_results: 客户端训练结果列表，每个元素包含client_id, loss, accuracy
            client_eval_results: 客户端评估结果列表（可选）
        """
        # 记录全局结果
        self.training_results["global_results"]["rounds"].append(round_num)
        self.training_results["global_results"]["global_loss"].append(global_loss)
        self.training_results["global_results"]["global_accuracy"].append(global_acc)
        
        # 记录客户端结果
        for client_result in client_results:
            client_id = f"client_{client_result['client_id']}"
            if client_id not in self.training_results["client_results"]:
                self.training_results["client_results"][client_id] = {
                    "loss": [],
                    "accuracy": []
                }
            
            self.training_results["client_results"][client_id]["loss"].append(
                client_result["loss"]
            )
            self.training_results["client_results"][client_id]["accuracy"].append(
                client_result["accuracy"]
            )
        
        # 记录到日志文件
        self.logger.info(f"轮次 {round_num}: 全局损失={global_loss:.4f}, "
                        f"全局准确率={global_acc:.4f}")
        
        for client_result in client_results:
            self.logger.info(f"  客户端 {client_result['client_id']}: "
                           f"损失={client_result['loss']:.4f}, "
                           f"准确率={client_result['accuracy']:.4f}")
    
    def save_results(self, filename: str = None) -> str:
        """
        保存训练结果到文件
        
        Args:
            filename: 结果文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.training_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"训练结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存结果文件失败: {e}")
            raise
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        从文件加载训练结果
        
        Args:
            filename: 结果文件名或完整路径
            
        Returns:
            训练结果字典
        """
        # 如果是相对路径，则在results_dir中查找
        if not os.path.isabs(filename):
            filepath = self.results_dir / filename
        else:
            filepath = Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"结果文件不存在: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.logger.info(f"成功加载结果文件: {filepath}")
            return results
            
        except Exception as e:
            self.logger.error(f"加载结果文件失败: {e}")
            raise
    
    def get_results_files(self) -> List[str]:
        """
        获取结果目录中的所有结果文件
        
        Returns:
            结果文件名列表
        """
        try:
            json_files = list(self.results_dir.glob("*.json"))
            return [f.name for f in json_files]
        except Exception as e:
            self.logger.error(f"获取结果文件列表失败: {e}")
            return []
    
    def get_results_files_with_info(self) -> List[Dict[str, Any]]:
        """
        获取结果目录中的所有结果文件及其基本信息
        
        Returns:
            包含文件信息的字典列表，每个字典包含filename, size, modified_time, experiment_info
        """
        try:
            json_files = list(self.results_dir.glob("*.json"))
            files_info = []
            
            for file_path in json_files:
                try:
                    # 获取文件基本信息
                    stat = file_path.stat()
                    file_info = {
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "experiment_info": {}
                    }
                    
                    # 尝试读取实验信息
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if "experiment_info" in data:
                                file_info["experiment_info"] = data["experiment_info"]
                    except:
                        # 如果读取失败，保持空的experiment_info
                        pass
                    
                    files_info.append(file_info)
                    
                except Exception as e:
                    self.logger.warning(f"获取文件 {file_path.name} 信息失败: {e}")
                    continue
            
            # 按修改时间排序（最新的在前）
            files_info.sort(key=lambda x: x["modified_time"], reverse=True)
            return files_info
            
        except Exception as e:
            self.logger.error(f"获取结果文件信息失败: {e}")
            return []
    
    def delete_results_file(self, filename: str) -> bool:
        """
        删除指定的结果文件
        
        Args:
            filename: 要删除的文件名
            
        Returns:
            删除是否成功
        """
        try:
            filepath = self.results_dir / filename
            if filepath.exists():
                filepath.unlink()
                self.logger.info(f"成功删除结果文件: {filename}")
                return True
            else:
                self.logger.warning(f"结果文件不存在: {filename}")
                return False
        except Exception as e:
            self.logger.error(f"删除结果文件失败: {e}")
            return False
    
    def backup_results_file(self, filename: str, backup_suffix: str = None) -> str:
        """
        备份指定的结果文件
        
        Args:
            filename: 要备份的文件名
            backup_suffix: 备份文件后缀，如果为None则使用时间戳
            
        Returns:
            备份文件的路径，失败时返回空字符串
        """
        try:
            source_path = self.results_dir / filename
            if not source_path.exists():
                self.logger.error(f"源文件不存在: {filename}")
                return ""
            
            # 生成备份文件名
            if backup_suffix is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_suffix = f"backup_{timestamp}"
            
            name_parts = filename.rsplit('.', 1)
            if len(name_parts) == 2:
                backup_filename = f"{name_parts[0]}_{backup_suffix}.{name_parts[1]}"
            else:
                backup_filename = f"{filename}_{backup_suffix}"
            
            backup_path = self.results_dir / backup_filename
            
            # 复制文件
            import shutil
            shutil.copy2(source_path, backup_path)
            
            self.logger.info(f"成功备份结果文件: {filename} -> {backup_filename}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"备份结果文件失败: {e}")
            return ""
    
    def merge_results(self, filenames: List[str], output_filename: str) -> bool:
        """
        合并多个结果文件
        
        Args:
            filenames: 要合并的文件名列表
            output_filename: 输出文件名
            
        Returns:
            合并是否成功
        """
        try:
            merged_data = {
                "merged_info": {
                    "source_files": filenames,
                    "merge_time": datetime.datetime.now().isoformat(),
                    "num_experiments": len(filenames)
                },
                "experiments": []
            }
            
            for filename in filenames:
                try:
                    data = self.load_results(filename)
                    experiment_data = {
                        "source_file": filename,
                        "data": data
                    }
                    merged_data["experiments"].append(experiment_data)
                except Exception as e:
                    self.logger.warning(f"加载文件 {filename} 失败，跳过: {e}")
                    continue
            
            # 保存合并结果
            output_path = self.results_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"成功合并 {len(merged_data['experiments'])} 个结果文件到: {output_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"合并结果文件失败: {e}")
            return False
    
    def export_results_csv(self, filename: str, csv_filename: str = None) -> str:
        """
        将结果文件导出为CSV格式
        
        Args:
            filename: 源JSON文件名
            csv_filename: 输出CSV文件名，如果为None则自动生成
            
        Returns:
            CSV文件路径，失败时返回空字符串
        """
        try:
            # 加载结果数据
            data = self.load_results(filename)
            
            if csv_filename is None:
                name_parts = filename.rsplit('.', 1)
                csv_filename = f"{name_parts[0]}.csv"
            
            csv_path = self.results_dir / csv_filename
            
            # 准备CSV数据
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入表头
                headers = ['Round', 'Global_Loss', 'Global_Accuracy']
                
                # 添加客户端列
                if 'client_results' in data:
                    client_ids = sorted(data['client_results'].keys())
                    for client_id in client_ids:
                        headers.extend([f'{client_id}_Loss', f'{client_id}_Accuracy'])
                
                writer.writerow(headers)
                
                # 写入数据行
                if 'global_results' in data:
                    rounds = data['global_results'].get('rounds', [])
                    global_loss = data['global_results'].get('global_loss', [])
                    global_acc = data['global_results'].get('global_accuracy', [])
                    
                    for i, round_num in enumerate(rounds):
                        row = [
                            round_num,
                            global_loss[i] if i < len(global_loss) else '',
                            global_acc[i] if i < len(global_acc) else ''
                        ]
                        
                        # 添加客户端数据
                        if 'client_results' in data:
                            for client_id in client_ids:
                                client_data = data['client_results'][client_id]
                                client_loss = client_data.get('loss', [])
                                client_acc = client_data.get('accuracy', [])
                                
                                row.extend([
                                    client_loss[i] if i < len(client_loss) else '',
                                    client_acc[i] if i < len(client_acc) else ''
                                ])
                        
                        writer.writerow(row)
            
            self.logger.info(f"成功导出CSV文件: {csv_filename}")
            return str(csv_path)
            
        except Exception as e:
            self.logger.error(f"导出CSV文件失败: {e}")
            return ""
    
    def validate_results_file(self, filename: str) -> Dict[str, Any]:
        """
        验证结果文件的完整性和格式
        
        Args:
            filename: 要验证的文件名
            
        Returns:
            验证结果字典，包含is_valid, errors, warnings等信息
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        try:
            # 检查文件是否存在
            filepath = self.results_dir / filename
            if not filepath.exists():
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"文件不存在: {filename}")
                return validation_result
            
            # 获取文件信息
            stat = filepath.stat()
            validation_result["file_info"] = {
                "size": stat.st_size,
                "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            # 尝试加载JSON
            try:
                data = self.load_results(filename)
            except json.JSONDecodeError as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"JSON格式错误: {e}")
                return validation_result
            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"文件读取错误: {e}")
                return validation_result
            
            # 验证数据结构
            required_keys = ["experiment_info", "global_results", "client_results"]
            for key in required_keys:
                if key not in data:
                    validation_result["errors"].append(f"缺少必需字段: {key}")
                    validation_result["is_valid"] = False
            
            # 验证global_results结构
            if "global_results" in data:
                global_results = data["global_results"]
                required_global_keys = ["rounds", "global_loss", "global_accuracy"]
                for key in required_global_keys:
                    if key not in global_results:
                        validation_result["errors"].append(f"global_results缺少字段: {key}")
                        validation_result["is_valid"] = False
                
                # 检查数据长度一致性
                if all(key in global_results for key in required_global_keys):
                    lengths = [len(global_results[key]) for key in required_global_keys]
                    if len(set(lengths)) > 1:
                        validation_result["warnings"].append(
                            f"global_results中数组长度不一致: {dict(zip(required_global_keys, lengths))}"
                        )
            
            # 验证client_results结构
            if "client_results" in data:
                client_results = data["client_results"]
                for client_id, client_data in client_results.items():
                    if not isinstance(client_data, dict):
                        validation_result["errors"].append(f"客户端 {client_id} 数据格式错误")
                        validation_result["is_valid"] = False
                        continue
                    
                    required_client_keys = ["loss", "accuracy"]
                    for key in required_client_keys:
                        if key not in client_data:
                            validation_result["warnings"].append(f"客户端 {client_id} 缺少字段: {key}")
                        elif not isinstance(client_data[key], list):
                            validation_result["errors"].append(f"客户端 {client_id} 的 {key} 不是列表格式")
                            validation_result["is_valid"] = False
            
            self.logger.info(f"结果文件验证完成: {filename}, 有效性: {validation_result['is_valid']}")
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"验证过程出错: {e}")
            self.logger.error(f"验证结果文件失败: {e}")
        
        return validation_result
    
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def log_debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)
    
    def save_final_results(self, filename: str = None) -> str:
        """
        保存最终训练结果
        
        Args:
            filename: 结果文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        return self.save_results(filename)
    
    def save_partial_results(self, round_num: int, filename: str = None) -> str:
        """
        保存部分训练结果（用于中断恢复）
        
        Args:
            round_num: 当前轮次
            filename: 结果文件名，如果为None则自动生成
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"partial_results_round_{round_num}_{timestamp}.json"
        
        # 创建部分结果数据
        partial_results = {
            "experiment_info": self.training_results["experiment_info"].copy(),
            "global_results": {
                "rounds": self.training_results["global_results"]["rounds"].copy(),
                "global_loss": self.training_results["global_results"]["global_loss"].copy(),
                "global_accuracy": self.training_results["global_results"]["global_accuracy"].copy()
            },
            "client_results": {},
            "partial_info": {
                "interrupted_at_round": round_num,
                "save_time": datetime.datetime.now().isoformat(),
                "is_partial": True
            }
        }
        
        # 复制客户端结果
        for client_id, client_data in self.training_results["client_results"].items():
            partial_results["client_results"][client_id] = {
                "loss": client_data["loss"].copy(),
                "accuracy": client_data["accuracy"].copy()
            }
        
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"部分训练结果已保存到: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存部分结果文件失败: {e}")
            return ""