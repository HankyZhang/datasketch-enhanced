#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HNSW算法示例
============

这个文件展示了如何使用HNSW (Hierarchical Navigable Small World) 算法
进行高效的近似最近邻搜索。

"""

import numpy as np
from datasketch import HNSW


def basic_usage_example():
    """基本使用示例"""
    print("=== HNSW基本使用示例 ===")
    
    # 创建随机数据
    dimension = 50
    num_points = 1000
    data = np.random.random((num_points, dimension))
    
    print(f"创建了 {num_points} 个 {dimension} 维的随机向量")
    
    # 初始化HNSW索引
    def euclidean_distance(x, y):
        return np.linalg.norm(x - y)
    
    index = HNSW(
        distance_func=euclidean_distance,
        m=16,                    # 每层最大连接数
        ef_construction=200,     # 构建时搜索宽度
    )
    
    print("初始化HNSW索引完成")
    
    # 批量插入数据
    print("开始插入数据...")
    index.update({i: vector for i, vector in enumerate(data)})
    print(f"成功插入 {len(index)} 个数据点")
    
    # 搜索最近邻
    query_vector = data[0]  # 使用第一个向量作为查询
    k = 10
    
    print(f"\n搜索前 {k} 个最近邻...")
    neighbors = index.query(query_vector, k=k, ef=100)
    
    print("搜索结果:")
    for i, (key, distance) in enumerate(neighbors):
        print(f"  {i+1}. 键: {key}, 距离: {distance:.6f}")


def similarity_search_example():
    """相似性搜索示例"""
    print("\n=== 相似性搜索示例 ===")
    
    # 创建一些有结构的数据
    np.random.seed(42)
    
    # 创建三个聚类的数据
    cluster1 = np.random.normal([0, 0], 0.1, (100, 2))
    cluster2 = np.random.normal([2, 2], 0.1, (100, 2))
    cluster3 = np.random.normal([-2, 2], 0.1, (100, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    labels = ['cluster1'] * 100 + ['cluster2'] * 100 + ['cluster3'] * 100
    
    print("创建了3个聚类的数据")
    
    # 使用余弦距离
    def cosine_distance(x, y):
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
    index = HNSW(distance_func=cosine_distance, m=8, ef_construction=100)
    
    # 插入数据
    for i, vector in enumerate(data):
        index.insert(labels[i] + f"_{i}", vector)
    
    print(f"插入了 {len(index)} 个带标签的数据点")
    
    # 搜索每个聚类的中心点
    test_points = [
        ("cluster1中心", np.array([0, 0])),
        ("cluster2中心", np.array([2, 2])),
        ("cluster3中心", np.array([-2, 2]))
    ]
    
    for name, query_point in test_points:
        print(f"\n查询 {name}:")
        neighbors = index.query(query_point, k=5, ef=50)
        for i, (key, distance) in enumerate(neighbors):
            cluster = key.split('_')[0]
            print(f"  {i+1}. {key} ({cluster}), 距离: {distance:.4f}")


def dynamic_operations_example():
    """动态操作示例"""
    print("\n=== 动态操作示例 ===")
    
    # 创建初始数据
    data = np.random.random((50, 10))
    
    index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
    
    # 逐个插入
    print("逐个插入数据点...")
    for i in range(20):
        index.insert(f"point_{i}", data[i])
    
    print(f"当前索引大小: {len(index)}")
    
    # 查询
    query = data[0]
    neighbors = index.query(query, k=3)
    print(f"查询结果: {len(neighbors)} 个邻居")
    
    # 更新一个点
    print("\n更新一个数据点...")
    new_vector = np.random.random(10)
    index.insert("point_0", new_vector)  # 更新已存在的点
    
    # 再次查询看变化
    neighbors_after = index.query(query, k=3)
    print("更新后的查询结果:")
    for key, distance in neighbors_after:
        print(f"  {key}: {distance:.4f}")
    
    # 删除一些点
    print("\n删除一些数据点...")
    index.remove("point_1")  # 软删除
    index.remove("point_2", hard=True)  # 硬删除
    
    print(f"删除后索引大小: {len(index)}")
    
    # 清理所有软删除的点
    print("清理软删除的点...")
    index.clean()
    print(f"清理后索引大小: {len(index)}")


def parameter_tuning_example():
    """参数调优示例"""
    print("\n=== 参数调优示例 ===")
    
    data = np.random.random((500, 20))
    
    # 不同参数配置的比较
    configs = [
        {"name": "快速配置", "m": 8, "ef_construction": 100, "ef": 50},
        {"name": "平衡配置", "m": 16, "ef_construction": 200, "ef": 100},
        {"name": "高精度配置", "m": 32, "ef_construction": 400, "ef": 200},
    ]
    
    for config in configs:
        print(f"\n测试 {config['name']}:")
        print(f"  参数: m={config['m']}, ef_construction={config['ef_construction']}")
        
        # 创建索引
        index = HNSW(
            distance_func=lambda x, y: np.linalg.norm(x - y),
            m=config['m'],
            ef_construction=config['ef_construction']
        )
        
        # 插入数据并测试
        import time
        start_time = time.time()
        index.update({i: vector for i, vector in enumerate(data)})
        build_time = time.time() - start_time
        
        # 测试查询性能
        query = data[0]
        start_time = time.time()
        neighbors = index.query(query, k=10, ef=config['ef'])
        query_time = time.time() - start_time
        
        print(f"  构建时间: {build_time:.3f}s")
        print(f"  查询时间: {query_time:.6f}s")
        print(f"  找到邻居: {len(neighbors)} 个")


def main():
    """运行所有示例"""
    print("HNSW算法演示")
    print("=" * 50)
    
    try:
        basic_usage_example()
        similarity_search_example()
        dynamic_operations_example()
        parameter_tuning_example()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("更多详细信息请参考文档：")
        print("- HNSW算法原理详解.md")
        print("- HNSW_代码分析_中文版.md")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
