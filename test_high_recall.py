#!/usr/bin/env python3
"""
Test Hybrid HNSW parameters for high recall with n_probe=10
"""

import numpy as np
import struct
import time
import json
from typing import Dict, List, Tuple, Optional

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

def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute ground truth using brute force."""
    print(f"Computing ground truth for {len(query_vectors)} queries...")
    
    ground_truth = []
    for i, query in enumerate(query_vectors):
        distances = np.linalg.norm(base_vectors - query, axis=1)
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        ground_truth.append(nearest_indices)
    
    return np.array(ground_truth)

def test_configuration(base_vectors, query_vectors, ground_truth, 
                      m, ef_construction, parent_level, k_children, 
                      parent_child_method='exact', n_probe=10):
    """Test a specific configuration with n_probe=10."""
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    print(f"\nTesting: m={m}, ef_c={ef_construction}, level={parent_level}, k_children={k_children}, method={parent_child_method}")
    
    # Build base index
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
    
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    start_time = time.time()
    base_index.update(dataset)
    base_build_time = time.time() - start_time
    
    # Build hybrid structure
    start_time = time.time()
    try:
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=parent_level,
            k_children=k_children,
            parent_child_method=parent_child_method
        )
        hybrid_build_time = time.time() - start_time
    except Exception as e:
        print(f"  ❌ Failed to build hybrid: {e}")
        return None
    
    stats = hybrid_index.get_stats()
    print(f"  Built: {stats.get('num_parents', 0)} parents, {stats.get('num_children', 0)} children")
    print(f"  Build time: {base_build_time + hybrid_build_time:.2f}s")
    
    # Test with n_probe=10
    query_times = []
    recalls_at_k = {1: [], 10: [], 100: []}
    
    for i, query in enumerate(query_vectors):
        start_time = time.time()
        try:
            search_results = hybrid_index.search(query, k=100, n_probe=n_probe)
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
            return None
            
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        result_ids = [rid for rid, _ in search_results]
        gt_ids = ground_truth[i]
        
        for k in [1, 10, 100]:
            k_min = min(k, len(result_ids), len(gt_ids))
            if k_min > 0:
                recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                recalls_at_k[k].append(recall_k)
    
    avg_query_time = np.mean(query_times) * 1000
    avg_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
    
    print(f"  Query time: {avg_query_time:.2f}ms")
    print(f"  Recall@1: {avg_recalls[1]:.4f}")
    print(f"  Recall@10: {avg_recalls[10]:.4f}")
    print(f"  Recall@100: {avg_recalls[100]:.4f}")
    
    return {
        'config': {
            'm': m,
            'ef_construction': ef_construction,
            'parent_level': parent_level,
            'k_children': k_children,
            'parent_child_method': parent_child_method,
            'n_probe': n_probe
        },
        'build_time': base_build_time + hybrid_build_time,
        'query_time_ms': avg_query_time,
        'recalls': avg_recalls,
        'stats': stats
    }

def main():
    print("🎯 High Recall Testing with n_probe=10")
    print("=" * 50)
    
    # Load smaller dataset for faster testing
    print("Loading SIFT data subset...")
    base_vectors = read_fvecs("sift/sift_base.fvecs", 3000)  # Smaller for faster testing
    query_vectors = read_fvecs("sift/sift_query.fvecs", 50)
    ground_truth = compute_ground_truth(base_vectors, query_vectors, k=100)
    print(f"Loaded {len(base_vectors)} base vectors, {len(query_vectors)} queries")
    
    # Test configurations targeting high recall
    test_configs = [
        # Standard configuration (baseline)
        (16, 200, 1, 1000, 'exact'),
        
        # Higher quality base index
        (16, 400, 1, 1000, 'exact'),
        (16, 600, 1, 1000, 'exact'),
        
        # Higher connectivity
        (32, 200, 1, 1000, 'exact'),
        (32, 400, 1, 1000, 'exact'),
        
        # More children per parent
        (16, 200, 1, 2000, 'exact'),
        (16, 400, 1, 2000, 'exact'),
        
        # Use level 0 (all nodes as parents)
        (16, 200, 0, 500, 'exact'),
        (16, 400, 0, 500, 'exact'),
        
        # Use level 0 with more children  
        (16, 200, 0, 1000, 'exact'),
        (16, 400, 0, 1000, 'exact'),
        
        # High quality configuration
        (32, 600, 1, 2000, 'exact'),
    ]
    
    results = []
    best_recall = 0
    best_config = None
    
    for m, ef_construction, parent_level, k_children, method in test_configs:
        result = test_configuration(
            base_vectors, query_vectors, ground_truth,
            m, ef_construction, parent_level, k_children, method, n_probe=10
        )
        
        if result:
            results.append(result)
            
            # Track best recall@10
            if result['recalls'][10] > best_recall:
                best_recall = result['recalls'][10]
                best_config = result
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 RESULTS ANALYSIS")
    print("=" * 50)
    
    # Sort by recall@10
    results.sort(key=lambda x: x['recalls'][10], reverse=True)
    
    print(f"{'Rank':<4} {'Recall@10':<10} {'Query Time':<12} {'Config':<40}")
    print("-" * 80)
    
    for i, result in enumerate(results[:10]):  # Top 10
        config = result['config']
        config_str = f"m={config['m']}, ef_c={config['ef_construction']}, level={config['parent_level']}, k_children={config['k_children']}"
        print(f"{i+1:<4} {result['recalls'][10]:<10.4f} {result['query_time_ms']:<12.2f} {config_str:<40}")
    
    print(f"\n🏆 BEST CONFIGURATION for Recall@10:")
    if best_config:
        config = best_config['config']
        print(f"  Parameters: m={config['m']}, ef_construction={config['ef_construction']}")
        print(f"              parent_level={config['parent_level']}, k_children={config['k_children']}")
        print(f"              method={config['parent_child_method']}")
        print(f"  Performance: Recall@10={best_config['recalls'][10]:.4f}")
        print(f"               Query time={best_config['query_time_ms']:.2f}ms")
        print(f"               Build time={best_config['build_time']:.2f}s")
        
        # Check if we achieved high recall (>0.9)
        if best_config['recalls'][10] >= 0.9:
            print(f"\n✅ HIGH RECALL ACHIEVED! {best_config['recalls'][10]:.1%} recall@10")
        elif best_config['recalls'][10] >= 0.8:
            print(f"\n🟡 GOOD RECALL: {best_config['recalls'][10]:.1%} recall@10")
        else:
            print(f"\n🔴 LOW RECALL: {best_config['recalls'][10]:.1%} recall@10 - consider:")
            print("    - Using parent_level=0 (all nodes)")
            print("    - Increasing ef_construction further")
            print("    - Increasing k_children")
            print("    - Using higher m value")
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_parameters': {'n_probe': 10},
        'dataset_info': {
            'base_vectors': len(base_vectors),
            'query_vectors': len(query_vectors)
        },
        'results': results,
        'best_config': best_config
    }
    
    with open('high_recall_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📁 Detailed results saved to high_recall_test_results.json")
    
    # Recommendations based on results
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 30)
    
    high_recall_configs = [r for r in results if r['recalls'][10] >= 0.9]
    good_recall_configs = [r for r in results if 0.8 <= r['recalls'][10] < 0.9]
    
    if high_recall_configs:
        fastest_high = min(high_recall_configs, key=lambda x: x['query_time_ms'])
        print(f"🚀 For >90% recall: Use config from rank {results.index(fastest_high) + 1}")
        
    if good_recall_configs:
        fastest_good = min(good_recall_configs, key=lambda x: x['query_time_ms'])
        print(f"⚡ For >80% recall: Use config from rank {results.index(fastest_good) + 1}")
    
    if not high_recall_configs and not good_recall_configs:
        print("💡 To achieve higher recall, try:")
        print("   - parent_level=0 with k_children=2000+")
        print("   - ef_construction=800+")
        print("   - m=64 for higher connectivity")
        print("   - Increase n_probe to 15-20")

if __name__ == "__main__":
    import os
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found!")
        exit(1)
    
    main()
