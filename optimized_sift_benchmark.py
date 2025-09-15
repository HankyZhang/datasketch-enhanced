#!/usr/bin/env python3
"""
Optimized SIFT Benchmark using tuned parameters for Hybrid HNSW
"""

import numpy as np
import struct
import time
import os
from typing import Dict, List, Tuple, Optional
import json

def read_fvecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .fvecs format files efficiently."""
    vectors = []
    count = 0
    
    with open(filename, 'rb') as f:
        while True:
            if max_count and count >= max_count:
                break
                
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
                
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
            
            if count % 10000 == 0:
                print(f"  Loaded {count} vectors...")
    
    return np.array(vectors, dtype=np.float32)

def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute ground truth using brute force (for subset testing)."""
    print(f"Computing ground truth for {len(query_vectors)} queries against {len(base_vectors)} base vectors...")
    
    ground_truth = []
    for i, query in enumerate(query_vectors):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(query_vectors)}")
            
        # Compute distances to all base vectors
        distances = np.linalg.norm(base_vectors - query, axis=1)
        
        # Get k nearest neighbors
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        
        ground_truth.append(nearest_indices)
    
    return np.array(ground_truth)

def optimized_sift_benchmark():
    """Run optimized SIFT benchmark with tuned parameters."""
    
    print("🚀 Optimized SIFT Benchmark - Tuned Parameters")
    print("=" * 60)
    
    # Load data (5K for comparison with previous results)
    print("Loading SIFT dataset...")
    base_vectors = read_fvecs("sift/sift_base.fvecs", 5000)
    query_vectors = read_fvecs("sift/sift_query.fvecs", 100)
    ground_truth = compute_ground_truth(base_vectors, query_vectors, k=100)
    
    print(f"Dataset loaded: {len(base_vectors)} base, {len(query_vectors)} queries")
    print()
    
    # Import modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    # Standard HNSW baseline
    print("=== Standard HNSW Baseline ===")
    start_time = time.time()
    standard_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    standard_index.update(dataset)
    standard_build_time = time.time() - start_time
    print(f"Build time: {standard_build_time:.2f}s")
    
    # Test standard with ef=100 (good balance)
    print("Testing standard HNSW (ef=100)...")
    query_times = []
    recalls_at_k = {1: [], 10: [], 100: []}
    
    for i, query in enumerate(query_vectors):
        start_time = time.time()
        search_results = standard_index.query(query, k=100, ef=100)
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        result_ids = [rid for rid, _ in search_results]
        gt_ids = ground_truth[i]
        
        for k in [1, 10, 100]:
            k_min = min(k, len(result_ids), len(gt_ids))
            if k_min > 0:
                recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                recalls_at_k[k].append(recall_k)
    
    standard_query_time = np.mean(query_times) * 1000
    standard_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
    
    print(f"Query time: {standard_query_time:.2f}ms")
    print(f"Recall@1: {standard_recalls[1]:.4f}")
    print(f"Recall@10: {standard_recalls[10]:.4f}")
    print()
    
    # Optimized Hybrid HNSW configurations
    configs = [
        {"name": "Speed Optimized", "parent_level": 1, "k_children": 1000, "n_probe": 5, "target": "0.75+ recall"},
        {"name": "Balanced", "parent_level": 1, "k_children": 1000, "n_probe": 10, "target": "0.82+ recall"},
        {"name": "Quality Optimized", "parent_level": 1, "k_children": 500, "n_probe": 20, "target": "0.90+ recall"}
    ]
    
    results = {}
    
    for config in configs:
        print(f"=== Hybrid HNSW - {config['name']} ===")
        print(f"Target: {config['target']}")
        print(f"Parameters: parent_level={config['parent_level']}, k_children={config['k_children']}, n_probe={config['n_probe']}")
        
        # Build base index (reuse for each config)
        base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
        base_index.update(dataset)
        
        # Build hybrid structure
        start_time = time.time()
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=config['parent_level'],
            k_children=config['k_children'],
            parent_child_method='exact'
        )
        hybrid_build_time = time.time() - start_time
        total_build_time = standard_build_time + hybrid_build_time
        
        stats = hybrid_index.get_stats()
        print(f"Build time: {total_build_time:.2f}s (base: {standard_build_time:.2f}s + hybrid: {hybrid_build_time:.2f}s)")
        print(f"Structure: {stats.get('num_parents', 0)} parents, {stats.get('num_children', 0)} children")
        
        # Test with optimal n_probe
        print(f"Testing with n_probe={config['n_probe']}...")
        query_times = []
        recalls_at_k = {1: [], 10: [], 100: []}
        
        for i, query in enumerate(query_vectors):
            start_time = time.time()
            search_results = hybrid_index.search(query, k=100, n_probe=config['n_probe'])
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            result_ids = [rid for rid, _ in search_results]
            gt_ids = ground_truth[i]
            
            for k in [1, 10, 100]:
                k_min = min(k, len(result_ids), len(gt_ids))
                if k_min > 0:
                    recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                    recalls_at_k[k].append(recall_k)
        
        hybrid_query_time = np.mean(query_times) * 1000
        hybrid_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
        
        print(f"Query time: {hybrid_query_time:.2f}ms")
        print(f"Recall@1: {hybrid_recalls[1]:.4f}")
        print(f"Recall@10: {hybrid_recalls[10]:.4f}")
        
        speedup = standard_query_time / hybrid_query_time
        print(f"Speedup: {speedup:.2f}x vs standard")
        print()
        
        results[config['name']] = {
            'config': config,
            'build_time': total_build_time,
            'query_time_ms': hybrid_query_time,
            'recalls': hybrid_recalls,
            'speedup': speedup,
            'stats': stats
        }
    
    # Summary comparison
    print("=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Query Time':<12} {'Recall@10':<12} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Standard HNSW':<20} {standard_query_time:<12.2f} {standard_recalls[10]:<12.4f} {'1.00x':<10}")
    
    for name, result in results.items():
        print(f"{name:<20} {result['query_time_ms']:<12.2f} {result['recalls'][10]:<12.4f} {result['speedup']:<10.2f}")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    print("-" * 30)
    
    # Find best for each use case
    best_speed = min(results.values(), key=lambda x: x['query_time_ms'])
    best_recall = max(results.values(), key=lambda x: x['recalls'][10])
    best_balanced = min([r for r in results.values() if r['recalls'][10] >= 0.8], 
                       key=lambda x: x['query_time_ms'], default=None)
    
    print(f"🏃 Fastest: {[k for k, v in results.items() if v == best_speed][0]}")
    print(f"   {best_speed['query_time_ms']:.2f}ms, {best_speed['speedup']:.1f}x speedup, {best_speed['recalls'][10]:.1%} recall")
    
    print(f"🎯 Best Recall: {[k for k, v in results.items() if v == best_recall][0]}")
    print(f"   {best_recall['query_time_ms']:.2f}ms, {best_recall['speedup']:.1f}x speedup, {best_recall['recalls'][10]:.1%} recall")
    
    if best_balanced:
        print(f"⚖️  Best Balanced: {[k for k, v in results.items() if v == best_balanced][0]}")
        print(f"   {best_balanced['query_time_ms']:.2f}ms, {best_balanced['speedup']:.1f}x speedup, {best_balanced['recalls'][10]:.1%} recall")
    
    # Save results
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'base_vectors': len(base_vectors),
            'query_vectors': len(query_vectors),
            'dimensions': base_vectors.shape[1]
        },
        'standard_hnsw': {
            'build_time': standard_build_time,
            'query_time_ms': standard_query_time,
            'recalls': standard_recalls
        },
        'hybrid_configurations': results
    }
    
    with open('optimized_sift_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📁 Results saved to optimized_sift_results.json")
    print("✅ Optimized benchmark completed!")

if __name__ == "__main__":
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found!")
        print("Please ensure the SIFT dataset is available in the 'sift' directory.")
        exit(1)
    
    optimized_sift_benchmark()
