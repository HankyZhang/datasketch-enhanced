#!/usr/bin/env python3
"""
Coverage演示脚本 - 展示repair功能对coverage的影响
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_hnsw.hnsw_hybrid import HybridHNSW

def test_coverage_scenarios():
    """测试不同场景下的coverage"""
    print("=" * 60)
    print("Coverage测试演示")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    data = np.random.randn(500, 128).astype(np.float32)
    
    print(f"数据集大小: {len(data)} 向量, 维度: {data.shape[1]}")
    print()
    
    # 场景1: 没有repair功能
    print("🔍 场景1: 没有repair功能")
    print("-" * 40)
    
    try:
        hybrid = HybridHNSW(
            space='l2',
            max_elements=len(data),
            base_hnsw_params={
                'M': 16,
                'ef_construction': 50,
                'max_m': 16,
                'max_m_l': 16
            },
            adaptive_config={
                'level': 2,
                'k_children': 15,  # 较小的k_children
                'approx_ef': 30,
                'repair_min_assignments': None,  # 不启用repair
                'diversify_max_assignments': None
            }
        )
        
        # 构建索引
        hybrid.add_items(data)
        
        # 获取统计信息
        stats = hybrid.get_mapping_diagnostics()
        coverage = stats.get('coverage_fraction', 0)
        num_parents = stats.get('num_parents', 0)
        
        print(f"Parent数量: {num_parents}")
        print(f"K_children: 15")
        print(f"Coverage: {coverage:.4f}")
        
        if coverage < 1.0:
            print(f"❌ Coverage < 1.0 - 有 {int((1-coverage)*len(data))} 个节点未被覆盖")
        else:
            print(f"✅ Coverage = 1.0 - 所有节点都被覆盖")
            
    except Exception as e:
        print(f"错误: {e}")
    
    print()
    
    # 场景2: 启用repair功能
    print("🔧 场景2: 启用repair功能")
    print("-" * 40)
    
    try:
        hybrid_repair = HybridHNSW(
            space='l2',
            max_elements=len(data),
            base_hnsw_params={
                'M': 16,
                'ef_construction': 50,
                'max_m': 16,
                'max_m_l': 16
            },
            adaptive_config={
                'level': 2,
                'k_children': 15,  # 同样的k_children
                'approx_ef': 30,
                'repair_min_assignments': 2,  # 启用repair
                'diversify_max_assignments': None
            }
        )
        
        # 构建索引
        hybrid_repair.add_items(data)
        
        # 获取统计信息
        stats_repair = hybrid_repair.get_mapping_diagnostics()
        coverage_repair = stats_repair.get('coverage_fraction', 0)
        num_parents_repair = stats_repair.get('num_parents', 0)
        
        print(f"Parent数量: {num_parents_repair}")
        print(f"K_children: 15")
        print(f"Repair启用: min_assignments=2")
        print(f"Coverage: {coverage_repair:.4f}")
        
        if coverage_repair < 1.0:
            print(f"❌ Coverage < 1.0 - 有 {int((1-coverage_repair)*len(data))} 个节点未被覆盖")
        else:
            print(f"✅ Coverage = 1.0 - repair功能确保了完全覆盖")
            
    except Exception as e:
        print(f"错误: {e}")
    
    print()
    print("=" * 60)
    print("总结:")
    print("• Repair功能可以确保coverage=1.0")
    print("• 没有repair时，coverage可能<1.0，特别是：")
    print("  - Parent数量少")
    print("  - k_children参数小")
    print("  - 高level的HNSW节点稀疏")
    print("=" * 60)

if __name__ == "__main__":
    test_coverage_scenarios()
