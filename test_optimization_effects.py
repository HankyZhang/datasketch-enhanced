#!/usr/bin/env python3
"""
Quick test to demonstrate the effect of diversification and repair parameters
on hybrid HNSW coverage and performance.
"""

import numpy as np
import struct
import time
from typing import Dict, List, Optional

def read_fvecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .fvecs format files efficiently."""
    vectors = []
    count = 0
    
    with open(filename, 'rb') as f:
        while True:
            if max_count and count >= max_count:
                break
                
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
                
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
    
    return np.array(vectors, dtype=np.float32)

def test_optimization_effects():
    """Test the effects of diversify_max=3 and repair_min=1 on coverage."""
    print("🧪 Testing Optimization Effects: diversify_max=3, repair_min=1")
    print("=" * 70)
    
    # Load smaller dataset for quick testing
    print("Loading SIFT data subset...")
    base_vectors = read_fvecs("sift/sift_base.fvecs", 10000)  # 10K vectors
    print(f"Loaded {len(base_vectors)} base vectors")
    
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Build base index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    base_index.update(dataset)
    print("Base index built successfully")
    
    configurations = [
        {"name": "Baseline", "diversify_max": None, "repair_min": None},
        {"name": "With Diversify=3", "diversify_max": 3, "repair_min": None},
        {"name": "With Repair=1", "diversify_max": None, "repair_min": 1},
        {"name": "Optimized (div=3, repair=1)", "diversify_max": 3, "repair_min": 1},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n📊 Testing: {config['name']}")
        print("-" * 40)
        
        start_time = time.time()
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=2,
            k_children=500,
            parent_child_method='approx',
            approx_ef=2000,
            diversify_max_assignments=config['diversify_max'],
            repair_min_assignments=config['repair_min']
        )
        build_time = time.time() - start_time
        
        stats = hybrid_index.get_stats()
        
        result = {
            'name': config['name'],
            'diversify_max': config['diversify_max'],
            'repair_min': config['repair_min'],
            'build_time': build_time,
            'num_parents': stats.get('num_parents', 0),
            'num_children': stats.get('num_children', 0),
            'coverage_fraction': stats.get('coverage_fraction', 0),
            'avg_children_per_parent': stats.get('avg_children_per_parent', 0),
        }
        results.append(result)
        
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Parents: {result['num_parents']}")
        print(f"  Children: {result['num_children']}")
        print(f"  Coverage: {result['coverage_fraction']:.4f}")
        print(f"  Avg children/parent: {result['avg_children_per_parent']:.1f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📊 SUMMARY COMPARISON")
    print("=" * 70)
    
    baseline = results[0]
    print(f"{'Configuration':<25} {'Coverage':<10} {'Children':<10} {'Improvement':<12}")
    print("-" * 70)
    
    for result in results:
        coverage = result['coverage_fraction']
        children = result['num_children']
        improvement = f"+{children - baseline['num_children']}" if children > baseline['num_children'] else f"{children - baseline['num_children']}"
        print(f"{result['name']:<25} {coverage:<10.4f} {children:<10} {improvement:<12}")
    
    # Key insights
    print("\n🎯 KEY INSIGHTS:")
    optimized = results[-1]  # Last one should be fully optimized
    
    coverage_improvement = optimized['coverage_fraction'] - baseline['coverage_fraction']
    children_improvement = optimized['num_children'] - baseline['num_children']
    
    print(f"✅ Coverage improvement: +{coverage_improvement:.4f} ({coverage_improvement*100:.1f}%)")
    print(f"✅ Children coverage improvement: +{children_improvement} vectors")
    
    if children_improvement > 0:
        print(f"🎉 SUCCESS: Optimization algorithms work! {children_improvement} more vectors covered.")
    else:
        print(f"⚠️  Unexpected: No improvement in coverage. Check parameters or dataset size.")

if __name__ == "__main__":
    import os
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found! Please ensure sift/ directory exists.")
    else:
        test_optimization_effects()
