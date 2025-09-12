#!/usr/bin/env python3
"""
Comprehensive Baseline vs Hybrid HNSW Comparison

This script provides a direct comparison between baseline HNSW and our hybrid implementation
to demonstrate the recall improvements achieved by the two-stage system.
"""

import numpy as np
import time
import json
import sys
import os
from typing import List, Tuple, Dict, Any

# Import baseline HNSW
from datasketch.hnsw import HNSW

# Import our hybrid implementation
from hnsw_hybrid_evaluation import HybridHNSWIndex, generate_synthetic_dataset, create_query_set


def l2_distance(x, y):
    """Euclidean distance function."""
    return np.linalg.norm(x - y)


def compute_ground_truth(dataset: np.ndarray, queries: Dict[int, np.ndarray], k: int) -> Dict[int, List[int]]:
    """Compute ground truth using brute force search."""
    print(f"Computing ground truth for {len(queries)} queries with k={k}...")
    ground_truth = {}
    
    for query_id, query_vector in queries.items():
        distances = []
        for data_id, data_vector in enumerate(dataset):
            dist = l2_distance(query_vector, data_vector)
            distances.append((data_id, dist))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        ground_truth[query_id] = [idx for idx, _ in distances[:k]]
    
    return ground_truth


def calculate_recall(predicted_ids: List[int], ground_truth_ids: List[int]) -> float:
    """Calculate recall between predicted results and ground truth."""
    if not ground_truth_ids:
        return 0.0
    
    predicted_set = set(predicted_ids)
    gt_set = set(ground_truth_ids)
    
    intersection = predicted_set.intersection(gt_set)
    return len(intersection) / len(gt_set)


def test_baseline_hnsw(dataset: np.ndarray, queries: Dict[int, np.ndarray], k: int) -> Dict[str, Any]:
    """Test baseline HNSW performance."""
    print("\n🔍 Testing Baseline HNSW...")
    
    # Remove query points from dataset
    query_indices = set(queries.keys())
    filtered_dataset = []
    filtered_indices = []
    
    for i, vector in enumerate(dataset):
        if i not in query_indices:
            filtered_dataset.append(vector)
            filtered_indices.append(i)
    
    filtered_dataset = np.array(filtered_dataset)
    
    # Build baseline HNSW index
    print("  Building baseline HNSW index...")
    start_time = time.time()
    
    baseline_hnsw = HNSW(
        distance_func=l2_distance,
        m=16,
        ef_construction=200
    )
    
    for idx, vector in zip(filtered_indices, filtered_dataset):
        baseline_hnsw.insert(idx, vector)
    
    build_time = time.time() - start_time
    print(f"  Build time: {build_time:.2f}s")
    
    # Test different ef_search values
    ef_search_values = [50, 100, 200]
    results = []
    
    for ef_search in ef_search_values:
        print(f"  Testing ef_search={ef_search}...")
        
        start_time = time.time()
        recalls = []
        
        for query_id, query_vector in queries.items():
            # Get baseline HNSW results
            hnsw_results = baseline_hnsw.query(query_vector, k=k, ef=ef_search)
            hnsw_ids = [result_id for result_id, _ in hnsw_results]
            
            # We'll compute recall against ground truth later
            recalls.append(hnsw_ids)
        
        query_time = time.time() - start_time
        avg_query_time = query_time / len(queries)
        
        results.append({
            'ef_search': ef_search,
            'avg_query_time': avg_query_time,
            'build_time': build_time,
            'results': recalls
        })
        
        print(f"    Avg query time: {avg_query_time*1000:.2f}ms")
    
    return {
        'method': 'Baseline HNSW',
        'config': {'m': 16, 'ef_construction': 200},
        'results': results,
        'filtered_indices': filtered_indices
    }


def test_hybrid_hnsw(dataset: np.ndarray, queries: Dict[int, np.ndarray], k: int) -> Dict[str, Any]:
    """Test hybrid HNSW performance."""
    print("\n🚀 Testing Hybrid HNSW...")
    
    # Convert dataset to dictionary format and exclude queries
    query_indices = set(queries.keys())
    dataset_dict = {}
    
    for i, vector in enumerate(dataset):
        if i not in query_indices:
            dataset_dict[i] = vector
    
    # Build hybrid HNSW index
    print("  Building hybrid HNSW index...")
    start_time = time.time()
    
    hybrid_index = HybridHNSWIndex(k_children=1000, n_probe=15)
    
    # Build base index
    hybrid_index.build_base_index(dataset_dict)
    
    # Extract parent nodes and build mapping
    hybrid_index.extract_parent_nodes(target_level=2)
    hybrid_index.build_parent_child_mapping()
    
    build_time = time.time() - start_time
    print(f"  Build time: {build_time:.2f}s")
    
    # Test different n_probe values
    n_probe_values = [5, 10, 15, 20]
    results = []
    
    for n_probe in n_probe_values:
        print(f"  Testing n_probe={n_probe}...")
        
        # Update n_probe
        hybrid_index.n_probe = n_probe
        
        start_time = time.time()
        recalls = []
        
        for query_id, query_vector in queries.items():
            # Get hybrid HNSW results
            hybrid_results = hybrid_index.search(query_vector, k=k)
            hybrid_ids = [result_id for result_id, _ in hybrid_results]
            
            recalls.append(hybrid_ids)
        
        query_time = time.time() - start_time
        avg_query_time = query_time / len(queries)
        
        results.append({
            'n_probe': n_probe,
            'avg_query_time': avg_query_time,
            'build_time': build_time,
            'results': recalls
        })
        
        print(f"    Avg query time: {avg_query_time*1000:.2f}ms")
    
    return {
        'method': 'Hybrid HNSW',
        'config': {'k_children': 1000, 'target_level': 2},
        'results': results
    }


def compare_baseline_vs_hybrid(
    dataset_size: int = 10000,
    vector_dim: int = 128,
    n_queries: int = 100,
    k: int = 10
) -> Dict[str, Any]:
    """Compare baseline HNSW vs hybrid HNSW performance."""
    
    print("=" * 80)
    print("BASELINE vs HYBRID HNSW COMPARISON")
    print("=" * 80)
    print(f"Dataset size: {dataset_size}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Number of queries: {n_queries}")
    print(f"k: {k}")
    print()
    
    # Generate dataset and queries
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(dataset_size, vector_dim)
    queries = create_query_set(dataset, n_queries)
    
    # Compute ground truth
    ground_truth = compute_ground_truth(dataset, queries, k)
    
    # Test baseline HNSW
    baseline_results = test_baseline_hnsw(dataset, queries, k)
    
    # Test hybrid HNSW
    hybrid_results = test_hybrid_hnsw(dataset, queries, k)
    
    # Calculate recalls for both methods
    print("\n📊 CALCULATING RECALLS...")
    
    # Calculate baseline recalls
    for result in baseline_results['results']:
        recalls = []
        for query_idx, predicted_ids in enumerate(result['results']):
            # Map query index to actual query ID
            query_id = list(queries.keys())[query_idx]
            gt_ids = ground_truth[query_id]
            recall = calculate_recall(predicted_ids, gt_ids)
            recalls.append(recall)
        
        result['recall@k'] = np.mean(recalls)
        result['recall_std'] = np.std(recalls)
        
        print(f"  Baseline (ef_search={result['ef_search']}): Recall@{k} = {result['recall@k']:.4f} ± {result['recall_std']:.4f}")
    
    # Calculate hybrid recalls
    for result in hybrid_results['results']:
        recalls = []
        for query_idx, predicted_ids in enumerate(result['results']):
            # Map query index to actual query ID
            query_id = list(queries.keys())[query_idx]
            gt_ids = ground_truth[query_id]
            recall = calculate_recall(predicted_ids, gt_ids)
            recalls.append(recall)
        
        result['recall@k'] = np.mean(recalls)
        result['recall_std'] = np.std(recalls)
        
        print(f"  Hybrid (n_probe={result['n_probe']}): Recall@{k} = {result['recall@k']:.4f} ± {result['recall_std']:.4f}")
    
    # Find best configurations
    best_baseline = max(baseline_results['results'], key=lambda x: x['recall@k'])
    best_hybrid = max(hybrid_results['results'], key=lambda x: x['recall@k'])
    
    # Calculate improvement
    improvement = (best_hybrid['recall@k'] - best_baseline['recall@k']) / best_baseline['recall@k'] * 100
    
    print(f"\n🏆 COMPARISON SUMMARY:")
    print("-" * 50)
    print(f"Best Baseline HNSW:")
    print(f"  Recall@{k}: {best_baseline['recall@k']:.4f}")
    print(f"  Config: ef_search={best_baseline['ef_search']}")
    print(f"  Query time: {best_baseline['avg_query_time']*1000:.2f}ms")
    print()
    print(f"Best Hybrid HNSW:")
    print(f"  Recall@{k}: {best_hybrid['recall@k']:.4f}")
    print(f"  Config: n_probe={best_hybrid['n_probe']}")
    print(f"  Query time: {best_hybrid['avg_query_time']*1000:.2f}ms")
    print()
    print(f"📈 Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("✅ Hybrid system shows improved recall!")
    else:
        print("⚠️  Baseline performs better in this test")
    
    return {
        'dataset_size': dataset_size,
        'vector_dim': vector_dim,
        'n_queries': n_queries,
        'k': k,
        'baseline_results': baseline_results,
        'hybrid_results': hybrid_results,
        'best_baseline': best_baseline,
        'best_hybrid': best_hybrid,
        'improvement_percentage': improvement
    }


def main():
    """Main function to run the comparison."""
    
    # Test different scales
    test_configs = [
        {'dataset_size': 5000, 'vector_dim': 64, 'n_queries': 50, 'k': 10},
        {'dataset_size': 10000, 'vector_dim': 128, 'n_queries': 100, 'k': 10},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"Testing with dataset_size={config['dataset_size']}, dim={config['vector_dim']}")
        print(f"{'='*80}")
        
        results = compare_baseline_vs_hybrid(**config)
        all_results.append(results)
    
    # Save comprehensive results
    output_file = "baseline_vs_hybrid_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Comprehensive results saved to {output_file}")
    
    # Summary
    print(f"\n🎯 OVERALL SUMMARY:")
    print("-" * 50)
    
    for i, result in enumerate(all_results):
        config = test_configs[i]
        print(f"Test {i+1} (size={config['dataset_size']}):")
        print(f"  Best Baseline: {result['best_baseline']['recall@k']:.4f}")
        print(f"  Best Hybrid: {result['best_hybrid']['recall@k']:.4f}")
        print(f"  Improvement: {result['improvement_percentage']:+.1f}%")
    
    avg_improvement = np.mean([r['improvement_percentage'] for r in all_results])
    print(f"\nAverage improvement: {avg_improvement:+.1f}%")


if __name__ == "__main__":
    main()
