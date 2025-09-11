#!/usr/bin/env python3
"""
Simple Baseline HNSW Recall Test

This script provides a focused test to measure baseline HNSW recall performance
and establish clear benchmarks.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from datasketch.hnsw import HNSW


def l2_distance(x, y):
    """Euclidean distance function."""
    return np.linalg.norm(x - y)


def test_baseline_hnsw_recall():
    """Test baseline HNSW recall performance."""
    
    print("=" * 60)
    print("BASELINE HNSW RECALL TEST")
    print("=" * 60)
    
    # Test parameters
    n_data = 2000
    n_queries = 50
    dim = 64
    k = 10
    
    print(f"Dataset size: {n_data}")
    print(f"Query count: {n_queries}")
    print(f"Vector dimension: {dim}")
    print(f"k: {k}")
    print()
    
    # Generate synthetic data
    np.random.seed(42)
    dataset = np.random.random((n_data, dim)).astype(np.float32)
    
    # Select queries from dataset
    query_indices = np.random.choice(n_data, n_queries, replace=False)
    queries = dataset[query_indices]
    
    # Remove query points from dataset to avoid trivial matches
    data_mask = np.ones(n_data, dtype=bool)
    data_mask[query_indices] = False
    filtered_dataset = dataset[data_mask]
    filtered_indices = np.arange(n_data)[data_mask]
    
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print()
    
    # Compute ground truth using brute force
    print("Computing ground truth...")
    start_time = time.time()
    
    ground_truth = []
    for query in queries:
        distances = []
        for i, data_vector in enumerate(filtered_dataset):
            dist = l2_distance(query, data_vector)
            distances.append((filtered_indices[i], dist))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        gt_ids = [idx for idx, _ in distances[:k]]
        ground_truth.append(gt_ids)
    
    gt_time = time.time() - start_time
    print(f"Ground truth computed in {gt_time:.2f}s")
    print()
    
    # Test baseline HNSW with different configurations
    configs = [
        {'m': 8, 'ef_construction': 100, 'ef_search': 50},
        {'m': 16, 'ef_construction': 200, 'ef_search': 100},
        {'m': 32, 'ef_construction': 400, 'ef_search': 200},
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing: m={config['m']}, ef_construction={config['ef_construction']}, ef_search={config['ef_search']}")
        
        # Build HNSW index
        start_time = time.time()
        hnsw = HNSW(
            distance_func=l2_distance,
            m=config['m'],
            ef_construction=config['ef_construction']
        )
        
        # Insert filtered data
        for idx, vector in zip(filtered_indices, filtered_dataset):
            hnsw.insert(idx, vector)
        
        build_time = time.time() - start_time
        print(f"  Build time: {build_time:.2f}s")
        
        # Test queries
        start_time = time.time()
        recalls = []
        
        for i, query in enumerate(queries):
            # Get HNSW results
            hnsw_results = hnsw.query(query, k=k, ef=config['ef_search'])
            hnsw_ids = [result_id for result_id, _ in hnsw_results]
            
            # Calculate recall
            gt_set = set(ground_truth[i])
            hnsw_set = set(hnsw_ids)
            recall = len(gt_set.intersection(hnsw_set)) / len(gt_set) if gt_set else 0.0
            recalls.append(recall)
        
        query_time = time.time() - start_time
        avg_query_time = query_time / n_queries
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        
        result = {
            'config': config,
            'avg_recall': avg_recall,
            'std_recall': std_recall,
            'avg_query_time': avg_query_time,
            'build_time': build_time
        }
        results.append(result)
        
        print(f"  Recall@{k}: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"  Avg query time: {avg_query_time*1000:.2f}ms")
        print()
    
    # Summary
    print("📊 SUMMARY:")
    print("-" * 60)
    
    best_config = max(results, key=lambda x: x['avg_recall'])
    fastest_config = min(results, key=lambda x: x['avg_query_time'])
    
    print(f"Best Recall:")
    print(f"  Recall@{k}: {best_config['avg_recall']:.4f}")
    print(f"  Config: {best_config['config']}")
    print(f"  Query time: {best_config['avg_query_time']*1000:.2f}ms")
    print()
    
    print(f"Fastest Query:")
    print(f"  Recall@{k}: {fastest_config['avg_recall']:.4f}")
    print(f"  Config: {fastest_config['config']}")
    print(f"  Query time: {fastest_config['avg_query_time']*1000:.2f}ms")
    print()
    
    # Performance summary
    recalls = [r['avg_recall'] for r in results]
    query_times = [r['avg_query_time'] * 1000 for r in results]  # Convert to ms
    
    print(f"Recall range: {min(recalls):.4f} - {max(recalls):.4f}")
    print(f"Query time range: {min(query_times):.2f}ms - {max(query_times):.2f}ms")
    
    return results


if __name__ == "__main__":
    test_baseline_hnsw_recall()
