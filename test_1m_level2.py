#!/usr/bin/env python3
"""
Test 1M SIFT dataset with parent_level=2 and n_probe constraint
Constraint: n_probe <= 0.5 * number_of_parents
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
            
            if count % 100000 == 0:
                print(f"  Loaded {count} vectors...")
    
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .ivecs format files efficiently."""
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
                
            vector = struct.unpack('i' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
    
    return np.array(vectors, dtype=np.int32)

def test_1m_configuration(m, ef_construction, k_children, test_queries=100):
    """Test a specific configuration on 1M dataset with parent_level=2."""
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    print(f"\n{'='*60}")
    print(f"Testing Configuration:")
    print(f"  m = {m}")
    print(f"  ef_construction = {ef_construction}")
    print(f"  k_children = {k_children}")
    print(f"  parent_level = 2 (fixed)")
    print(f"{'='*60}")
    
    # Load full 1M dataset
    print("Loading 1M SIFT base vectors...")
    base_vectors = read_fvecs("sift/sift_base.fvecs")  # Full 1M
    print(f"Loaded {len(base_vectors)} base vectors")
    
    print(f"Loading {test_queries} query vectors...")
    query_vectors = read_fvecs("sift/sift_query.fvecs", test_queries)
    print(f"Loaded {len(query_vectors)} query vectors")
    
    print("Loading ground truth...")
    ground_truth = read_ivecs("sift/sift_groundtruth.ivecs", test_queries)
    print(f"Loaded ground truth for {len(ground_truth)} queries")
    
    # Build base HNSW index
    print("\n🔨 Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    start_time = time.time()
    base_index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
    
    # Build in batches to show progress
    batch_size = 50000
    for i in range(0, len(base_vectors), batch_size):
        end_i = min(i + batch_size, len(base_vectors))
        batch = {j: base_vectors[j] for j in range(i, end_i)}
        base_index.update(batch)
        print(f"  Progress: {end_i}/{len(base_vectors)} vectors processed")
    
    base_build_time = time.time() - start_time
    print(f"✅ Base HNSW built in {base_build_time:.2f}s")
    
    # Build hybrid structure
    print("\n🚀 Building hybrid structure (parent_level=2)...")
    start_time = time.time()
    
    try:
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=2,
            k_children=k_children,
            parent_child_method='exact'
        )
        hybrid_build_time = time.time() - start_time
        
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        num_children = stats.get('num_children', 0)
        avg_children = stats.get('avg_children_per_parent', 0)
        
        print(f"✅ Hybrid structure built in {hybrid_build_time:.2f}s")
        print(f"📊 Structure stats:")
        print(f"   Parents: {num_parents}")
        print(f"   Children: {num_children}")
        print(f"   Avg children/parent: {avg_children:.1f}")
        
        # Calculate n_probe constraint
        max_n_probe = max(1, int(0.5 * num_parents))
        print(f"🎯 n_probe constraint: n_probe ≤ {max_n_probe} (0.5 * {num_parents})")
        
        # Test different n_probe values within constraint
        n_probe_values = []
        for n_probe in [1, 2, 5, 10, 20, 50, 100]:
            if n_probe <= max_n_probe:
                n_probe_values.append(n_probe)
        
        if not n_probe_values:
            n_probe_values = [1]  # At least test with n_probe=1
        
        print(f"🧪 Testing n_probe values: {n_probe_values}")
        
        results = {}
        
        for n_probe in n_probe_values:
            print(f"\n📈 Testing n_probe = {n_probe}...")
            
            query_times = []
            recalls_at_k = {1: [], 10: [], 100: []}
            
            for i, query in enumerate(query_vectors):
                if i % 20 == 0:
                    print(f"  Query progress: {i}/{len(query_vectors)}")
                
                start_time = time.time()
                search_results = hybrid_index.search(query, k=100, n_probe=n_probe)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Calculate recall
                result_ids = [rid for rid, _ in search_results]
                gt_ids = ground_truth[i]
                
                for k in [1, 10, 100]:
                    k_min = min(k, len(result_ids), len(gt_ids))
                    if k_min > 0:
                        recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                        recalls_at_k[k].append(recall_k)
            
            avg_query_time = np.mean(query_times) * 1000  # Convert to ms
            avg_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
            
            results[n_probe] = {
                'query_time_ms': avg_query_time,
                'recalls': avg_recalls
            }
            
            print(f"  Results: {avg_query_time:.2f}ms, Recall@1={avg_recalls[1]:.4f}, Recall@10={avg_recalls[10]:.4f}")
        
        return {
            'config': {
                'm': m,
                'ef_construction': ef_construction,
                'k_children': k_children,
                'parent_level': 2
            },
            'build_times': {
                'base': base_build_time,
                'hybrid': hybrid_build_time,
                'total': base_build_time + hybrid_build_time
            },
            'stats': stats,
            'max_n_probe': max_n_probe,
            'results_by_n_probe': results,
            'dataset_size': len(base_vectors)
        }
        
    except Exception as e:
        print(f"❌ Failed to build hybrid structure: {e}")
        return None

def main():
    print("🎯 1M SIFT Dataset Parameter Testing")
    print("Constraints: parent_level=2, n_probe ≤ 0.5 * num_parents")
    print("=" * 70)
    
    # Parameter configurations to test
    # Focus on configurations that work well with level 2
    configs = [
        # (m, ef_construction, k_children)
        (16, 200, 2000),    # Standard with more children
        (16, 400, 2000),    # Higher quality base
        (32, 200, 2000),    # Higher connectivity  
        (16, 200, 5000),    # Many children per parent
        (32, 400, 3000),    # High quality + connectivity
        (16, 600, 2000),    # Very high quality base
    ]
    
    all_results = []
    
    for i, (m, ef_construction, k_children) in enumerate(configs, 1):
        print(f"\n🧪 Configuration {i}/{len(configs)}")
        
        try:
            result = test_1m_configuration(m, ef_construction, k_children, test_queries=100)
            if result:
                all_results.append(result)
                
                # Show quick summary
                best_n_probe = max(result['results_by_n_probe'].keys(), 
                                 key=lambda n: result['results_by_n_probe'][n]['recalls'][10])
                best_result = result['results_by_n_probe'][best_n_probe]
                
                print(f"\n✅ Best result for this config:")
                print(f"   n_probe = {best_n_probe}")
                print(f"   Query time = {best_result['query_time_ms']:.2f}ms")
                print(f"   Recall@10 = {best_result['recalls'][10]:.4f}")
                print(f"   Build time = {result['build_times']['total']:.2f}s")
                
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            continue
    
    # Analysis and recommendations
    if all_results:
        print(f"\n{'='*70}")
        print("📊 ANALYSIS & RECOMMENDATIONS")
        print(f"{'='*70}")
        
        # Find best configurations
        best_configs = []
        
        for result in all_results:
            config = result['config']
            
            # Find best n_probe for this config
            best_n_probe = max(result['results_by_n_probe'].keys(),
                             key=lambda n: result['results_by_n_probe'][n]['recalls'][10])
            best_perf = result['results_by_n_probe'][best_n_probe]
            
            best_configs.append({
                'config': config,
                'best_n_probe': best_n_probe,
                'max_n_probe': result['max_n_probe'],
                'num_parents': result['stats']['num_parents'],
                'query_time_ms': best_perf['query_time_ms'],
                'recall_10': best_perf['recalls'][10],
                'build_time': result['build_times']['total']
            })
        
        # Sort by recall@10
        best_configs.sort(key=lambda x: x['recall_10'], reverse=True)
        
        print(f"\n🏆 TOP CONFIGURATIONS (by Recall@10):")
        print(f"{'Rank':<4} {'Recall@10':<10} {'Query(ms)':<10} {'n_probe':<8} {'Parents':<8} {'Config':<25}")
        print("-" * 75)
        
        for i, config_result in enumerate(best_configs[:5], 1):
            config = config_result['config']
            config_str = f"m={config['m']}, ef_c={config['ef_construction']}, k_ch={config['k_children']}"
            
            print(f"{i:<4} {config_result['recall_10']:<10.4f} {config_result['query_time_ms']:<10.2f} "
                  f"{config_result['best_n_probe']:<8} {config_result['num_parents']:<8} {config_str:<25}")
        
        # Specific recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print("-" * 40)
        
        if best_configs:
            top_config = best_configs[0]
            print(f"🥇 Best Overall Configuration:")
            print(f"   m = {top_config['config']['m']}")
            print(f"   ef_construction = {top_config['config']['ef_construction']}")
            print(f"   k_children = {top_config['config']['k_children']}")
            print(f"   parent_level = 2")
            print(f"   Optimal n_probe = {top_config['best_n_probe']} (max allowed: {top_config['max_n_probe']})")
            print(f"   Expected Performance:")
            print(f"     - Recall@10: {top_config['recall_10']:.1%}")
            print(f"     - Query time: {top_config['query_time_ms']:.2f}ms")
            print(f"     - Build time: {top_config['build_time']:.0f}s")
            print(f"     - Parents: {top_config['num_parents']}")
        
        # Speed vs Accuracy tradeoff
        print(f"\n⚖️  Speed vs Accuracy Tradeoffs:")
        fastest = min(best_configs, key=lambda x: x['query_time_ms'])
        most_accurate = max(best_configs, key=lambda x: x['recall_10'])
        
        if fastest != most_accurate:
            print(f"   🏃 Fastest: m={fastest['config']['m']}, ef_c={fastest['config']['ef_construction']}")
            print(f"       {fastest['query_time_ms']:.2f}ms, {fastest['recall_10']:.1%} recall")
            print(f"   🎯 Most Accurate: m={most_accurate['config']['m']}, ef_c={most_accurate['config']['ef_construction']}")
            print(f"       {most_accurate['query_time_ms']:.2f}ms, {most_accurate['recall_10']:.1%} recall")
    
    # Save results
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': '1M SIFT',
        'constraints': {
            'parent_level': 2,
            'n_probe_constraint': '≤ 0.5 * num_parents'
        },
        'configurations_tested': len(configs),
        'successful_results': len(all_results),
        'results': all_results
    }
    
    filename = '1m_sift_level2_results.json'
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Detailed results saved to {filename}")
    print("✅ 1M dataset testing completed!")

if __name__ == "__main__":
    import os
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found!")
        print("Please ensure the sift/ directory exists with:")
        print("  - sift_base.fvecs (1M vectors)")
        print("  - sift_query.fvecs (10K queries)")  
        print("  - sift_groundtruth.ivecs (ground truth)")
        exit(1)
    
    main()
