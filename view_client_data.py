"""
查看保存的客户端数据
"""

import json
import pickle
import numpy as np
from data import DataSplitter


def view_meta_json(experiment_name):
    """查看meta.json文件内容"""
    import os
    
    meta_path = f"./client_data/{experiment_name}/meta.json"
    if not os.path.exists(meta_path):
        print(f"meta.json文件不存在: {meta_path}")
        return None
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    print(f"\n=== 实验元数据 ({experiment_name}) ===")
    
    # 实验信息
    exp_info = meta_data['experiment_info']
    print(f"实验名称: {exp_info['name']}")
    print(f"创建时间: {exp_info['created_time']}")
    print(f"客户端总数: {exp_info['total_clients']}")
    print(f"总样本数: {exp_info['total_samples']}")
    
    # 数据集信息
    dataset_info = meta_data['dataset_info']
    print(f"\n=== 数据集信息 ===")
    print(f"数据集名称: {dataset_info['name']}")
    print(f"类别数: {dataset_info['num_classes']}")
    print(f"训练样本数: {dataset_info['train_samples']}")
    print(f"测试样本数: {dataset_info['test_samples']}")
    print(f"数据形状: {dataset_info['data_shape']}")
    print(f"数据目录: {dataset_info['data_dir']}")
    
    # 分割配置
    split_config = meta_data['split_config']
    print(f"\n=== 分割配置 ===")
    print(f"分布类型: {split_config['distribution_type']}")
    print(f"客户端数量: {split_config['num_clients']}")
    if split_config['alpha'] is not None:
        print(f"Dirichlet参数α: {split_config['alpha']}")
    
    # 统计信息
    stats = meta_data['statistics']
    print(f"\n=== 统计信息 ===")
    print(f"平均每客户端样本数: {stats['avg_samples_per_client']:.1f}")
    print(f"样本数标准差: {stats['std_samples_per_client']:.1f}")
    print(f"最少样本数: {stats['min_samples_per_client']}")
    print(f"最多样本数: {stats['max_samples_per_client']}")
    
    print(f"\n=== 全局标签分布 ===")
    for label, count in sorted(stats['global_label_distribution'].items(), key=lambda x: int(x[0])):
        print(f"类别 {label}: {count} 个样本")
    
    # 客户端详细信息
    print(f"\n=== 客户端详细信息 ===")
    client_details = meta_data['client_details']
    
    print(f"{'客户端ID':<8} {'样本数':<8} {'类别数':<8} {'基尼系数':<10} {'主要类别':<15}")
    print("-" * 60)
    
    for client in client_details:
        client_id = client['client_id']
        num_samples = client['num_samples']
        num_classes = client['num_classes']
        gini = client['gini_coefficient']
        
        # 找出主要类别（样本数最多的前2个类别）
        label_dist = client['label_distribution']
        sorted_labels = sorted(label_dist.items(), key=lambda x: x[1], reverse=True)
        main_classes = f"{sorted_labels[0][0]}({sorted_labels[0][1]}), {sorted_labels[1][0]}({sorted_labels[1][1]})"
        
        print(f"{client_id:<8} {num_samples:<8} {num_classes:<8} {gini:<10.3f} {main_classes:<15}")
    
    return meta_data


def view_saved_client_data():
    """查看保存的客户端数据"""
    print("=== 查看保存的客户端数据 ===")
    
    # 初始化数据分割器
    splitter = DataSplitter()
    
    # 列出所有保存的实验
    experiments = splitter.list_saved_experiments()
    print(f"已保存的实验: {experiments}")
    
    if not experiments:
        print("没有找到保存的实验数据")
        return
    
    # 加载第一个实验的数据
    experiment_name = experiments[0]
    print(f"\n正在加载实验: {experiment_name}")
    
    try:
        client_data_list = splitter.load_client_data(experiment_name)
        print(f"成功加载 {len(client_data_list)} 个客户端的数据")
        
        # 获取统计信息
        stats = splitter.get_split_statistics(client_data_list)
        
        print(f"\n=== 数据分割统计信息 ===")
        print(f"客户端数量: {stats['num_clients']}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"平均每客户端样本数: {stats['avg_samples_per_client']:.1f}")
        print(f"样本数标准差: {stats['std_samples_per_client']:.1f}")
        print(f"最少样本数: {stats['min_samples_per_client']}")
        print(f"最多样本数: {stats['max_samples_per_client']}")
        
        print(f"\n=== 全局标签分布 ===")
        for label, count in sorted(stats['global_label_distribution'].items()):
            print(f"类别 {label}: {count} 个样本")
        
        print(f"\n=== 各客户端详细信息 ===")
        for i, client_data in enumerate(client_data_list):
            print(f"\n客户端 {i}:")
            print(f"  样本数: {client_data['num_samples']}")
            print(f"  数据形状: {client_data['data'].shape}")
            print(f"  标签形状: {client_data['labels'].shape}")
            
            # 显示标签分布
            unique_labels, counts = np.unique(client_data['labels'], return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"  标签分布: {label_dist}")
            
            # 显示数据统计
            print(f"  数据范围: [{client_data['data'].min():.3f}, {client_data['data'].max():.3f}]")
            print(f"  数据均值: {client_data['data'].mean():.3f}")
            print(f"  数据标准差: {client_data['data'].std():.3f}")
        
        # 演示如何使用单个客户端的数据
        print(f"\n=== 客户端数据使用示例 ===")
        client_0_data = client_data_list[0]
        print(f"客户端 0 的第一个样本:")
        print(f"  图像形状: {client_0_data['data'][0].shape}")
        print(f"  标签: {client_0_data['labels'][0]}")
        
    except Exception as e:
        print(f"加载数据时出错: {e}")


def load_specific_client_data(experiment_name, client_id):
    """加载特定客户端的数据"""
    print(f"\n=== 加载特定客户端数据 ===")
    
    import os
    client_file = f"./client_data/{experiment_name}/client_{client_id}.pkl"
    
    if not os.path.exists(client_file):
        print(f"客户端数据文件不存在: {client_file}")
        return None
    
    with open(client_file, 'rb') as f:
        client_data = pickle.load(f)
    
    print(f"客户端 {client_id} 数据:")
    print(f"  样本数: {client_data['num_samples']}")
    print(f"  数据形状: {client_data['data'].shape}")
    print(f"  标签形状: {client_data['labels'].shape}")
    
    return client_data


if __name__ == "__main__":
    # 首先查看meta.json文件
    print("=== 查看实验元数据 ===")
    splitter = DataSplitter()
    experiments = splitter.list_saved_experiments()
    
    if experiments:
        experiment_name = experiments[0]
        print(f"查看实验: {experiment_name}")
        view_meta_json(experiment_name)
    
    print("\n" + "="*80)
    
    # 然后查看详细的客户端数据
    view_saved_client_data()
    
    print("\n" + "="*80)
    
    # 演示加载特定客户端数据
    if experiments:
        load_specific_client_data(experiments[0], 0)