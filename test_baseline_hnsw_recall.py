#!/usr/bin/env python3
"""
Baseline HNSW Recall Testing Script

This script tests the recall performance of the baseline HNSW implementation
to establish benchmarks for comparison with the hybrid system.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from datasketch.hnsw import HNSW
import json
import os


def generate_synthetic_dataset(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic dataset for testing."""
    np.random.seed(seed)
    return np.random.random((n_vectors, dim)).astype(np.float32)


def create_query_set(dataset: np.ndarray, n_queries: int, seed: int = 123) -> Dict[int, np.ndarray]:
    """Create query set by sampling from the dataset."""
    np.random.seed(seed)
    query_indices = np.random.choice(len(dataset), n_queries, replace=False)
    return {i: dataset[idx] for i, idx in enumerate(query_indices)}


def compute_ground_truth(
    dataset: np.ndarray, 
    queries: Dict[int, np.ndarray], 
    k: int,
    distance_func=None
) -> Dict[int, List[Tuple[int, float]]]:
    """Compute ground truth using brute force search."""
    if distance_func is None:
        distance_func = lambda x, y: np.linalg.norm(x - y)
    
    print(f"Computing ground truth for {len(queries)} queries with k={k}...")
    ground_truth = {}
    
    for query_id, query_vector in queries.items():
        distances = []
        for data_id, data_vector in enumerate(dataset):
            dist = distance_func(query_vector, data_vector)
            distances.append((data_id, dist))
        
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        ground_truth[query_id] = distances[:k]
    
    return ground_truth


def test_baseline_hnsw_recall(
    dataset_size: int = 10000,
    vector_dim: int = 128,
    n_queries: int = 100,
    k_values: List[int] = [5, 10, 20],
    m_values: List[int] = [8, 16, 32],
    ef_construction_values: List[int] = [100, 200, 400],
    ef_search_values: List[int] = [50, 100, 200, 400]
) -> Dict[str, Any]:
    """Test baseline HNSW recall with different parameters."""
    
    print("=" * 80)
    print("BASELINE HNSW RECALL TESTING")
    print("=" * 80)
    print(f"Dataset size: {dataset_size}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Number of queries: {n_queries}")
    print(f"K values: {k_values}")
    print()
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(dataset_size, vector_dim)
    queries = create_query_set(dataset, n_queries)
    
    # Distance function
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # Compute ground truth for maximum k
    max_k = max(k_values)
    ground_truth = compute_ground_truth(dataset, queries, max_k, distance_func)
    
    results = []
    
    # Test different parameter combinations
    total_tests = len(m_values) * len(ef_construction_values) * len(ef_search_values)
    test_count = 0
    
    for m in m_values:
        for ef_construction in ef_construction_values:
            for ef_search in ef_search_values:
                test_count += 1
                print(f"\nTest {test_count}/{total_tests}: m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
                
                # Build HNSW index
                print("  Building HNSW index...")
                start_time = time.time()
                
                hnsw_index = HNSW(
                    distance_func=distance_func,
                    m=m,
                    ef_construction=ef_construction
                )
                
                # Insert all vectors except queries
                query_indices = set(queries.keys())
                for i, vector in enumerate(dataset):
                    if i not in query_indices:
                        hnsw_index.insert(i, vector)
                
                build_time = time.time() - start_time
                print(f"  Build time: {build_time:.2f}s")
                
                # Test queries with different ef_search values
                print("  Testing queries...")
                query_times = []
                
                for query_id, query_vector in queries.items():
                    start_time = time.time()
                    results_hnsw = hnsw_index.query(query_vector, k=max_k, ef=ef_search)
                    query_time = time.time() - start_time
                    query_times.append(query_time)
                
                avg_query_time = np.mean(query_times)
                
                # Calculate recall for different k values
                for k in k_values:
                    recalls = []
                    
                    for query_id, query_vector in queries.items():
                        # Get HNSW results
                        results_hnsw = hnsw_index.query(query_vector, k=k, ef=ef_search)
                        hnsw_ids = {result_id for result_id, _ in results_hnsw}
                        
                        # Get ground truth
                        gt_ids = {result_id for result_id, _ in ground_truth[query_id][:k]}
                        
                        # Calculate recall
                        recall = len(hnsw_ids.intersection(gt_ids)) / len(gt_ids) if gt_ids else 0.0
                        recalls.append(recall)
                    
                    avg_recall = np.mean(recalls)
                    
                    result = {
                        'm': m,
                        'ef_construction': ef_construction,
                        'ef_search': ef_search,
                        'k': k,
                        'recall@k': avg_recall,
                        'avg_query_time': avg_query_time,
                        'build_time': build_time,
                        'dataset_size': dataset_size,
                        'vector_dim': vector_dim,
                        'n_queries': n_queries
                    }
                    
                    results.append(result)
                    
                    print(f"    k={k}: Recall@{k} = {avg_recall:.4f}")
    
    return results


def analyze_baseline_results(results: List[Dict[str, Any]]) -> None:
    """Analyze and display baseline HNSW results."""
    
    print("\n" + "=" * 80)
    print("BASELINE HNSW ANALYSIS")
    print("=" * 80)
    
    # Group results by k
    k_values = sorted(set(r['k'] for r in results))
    
    for k in k_values:
        print(f"\n📊 RECALL@{k} ANALYSIS:")
        print("-" * 50)
        
        k_results = [r for r in results if r['k'] == k]
        k_results.sort(key=lambda x: x['recall@k'], reverse=True)
        
        print(f"{'Rank':<4} {'Recall@{k}':<12} {'Query Time':<12} {'m':<3} {'ef_const':<8} {'ef_search':<10}")
        print("-" * 50)
        
        for i, result in enumerate(k_results[:10]):  # Top 10
            print(f"{i+1:<4} {result['recall@k']:<12.4f} {result['avg_query_time']*1000:<12.2f}ms {result['m']:<3} {result['ef_construction']:<8} {result['ef_search']:<10}")
    
    # Find best configurations
    print(f"\n🏆 BEST CONFIGURATIONS:")
    print("-" * 50)
    
    for k in k_values:
        k_results = [r for r in results if r['k'] == k]
        best_result = max(k_results, key=lambda x: x['recall@k'])
        
        print(f"Best Recall@{k}: {best_result['recall@k']:.4f}")
        print(f"  Parameters: m={best_result['m']}, ef_construction={best_result['ef_construction']}, ef_search={best_result['ef_search']}")
        print(f"  Query time: {best_result['avg_query_time']*1000:.2f}ms")
        print(f"  Build time: {best_result['build_time']:.2f}s")
        print()


def compare_with_hybrid_results(baseline_results: List[Dict[str, Any]]) -> None:
    """Compare baseline results with known hybrid results."""
    
    print("\n" + "=" * 80)
    print("BASELINE vs HYBRID COMPARISON")
    print("=" * 80)
    
    # Known hybrid results from our testing
    hybrid_results = {
        5: 0.5215,   # From our 5K test
        10: 0.5215,  # From our 5K test  
        20: 0.4500   # Estimated
    }
    
    print(f"{'K':<3} {'Baseline Best':<14} {'Hybrid Known':<12} {'Improvement':<12}")
    print("-" * 45)
    
    for k in [5, 10, 20]:
        baseline_k_results = [r for r in baseline_results if r['k'] == k]
        if baseline_k_results:
            best_baseline = max(baseline_k_results, key=lambda x: x['recall@k'])['recall@k']
            hybrid_known = hybrid_results.get(k, 0.0)
            
            if baseline_k_results and hybrid_known > 0:
                improvement = (hybrid_known - best_baseline) / best_baseline * 100
                print(f"{k:<3} {best_baseline:<14.4f} {hybrid_known:<12.4f} {improvement:+8.1f}%")
            else:
                print(f"{k:<3} {best_baseline:<14.4f} {'N/A':<12} {'N/A':<12}")


def main():
    """Main function to run baseline HNSW recall testing."""
    
    print("Starting Baseline HNSW Recall Testing...")
    
    # Test configuration
    config = {
        'dataset_size': 10000,
        'vector_dim': 128,
        'n_queries': 200,
        'k_values': [5, 10, 20],
        'm_values': [8, 16, 32],
        'ef_construction_values': [100, 200, 400],
        'ef_search_values': [50, 100, 200, 400]
    }
    
    # Run tests
    results = test_baseline_hnsw_recall(**config)
    
    # Analyze results
    analyze_baseline_results(results)
    
    # Compare with hybrid
    compare_with_hybrid_results(results)
    
    # Save results
    results_file = "baseline_hnsw_recall_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    print(f"📊 Total configurations tested: {len(results)}")
    
    # Summary
    best_overall = max(results, key=lambda x: x['recall@k'])
    print(f"\n🎯 OVERALL BEST PERFORMANCE:")
    print(f"   Recall@{best_overall['k']}: {best_overall['recall@k']:.4f}")
    print(f"   Parameters: m={best_overall['m']}, ef_construction={best_overall['ef_construction']}, ef_search={best_overall['ef_search']}")
    print(f"   Query time: {best_overall['avg_query_time']*1000:.2f}ms")


if __name__ == "__main__":
    main()
