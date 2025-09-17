"""
Custom K-Means HNSW Parameter Tuning
"""

import sys
import os
sys.path.append('..')
from method3.tune_kmeans_hnsw import KMeansHNSWEvaluator, save_results, load_sift_data
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW
import numpy as np

# ========== CONFIGURATION ==========
# Data size settings
DATASET_SIZE = 5000      # Number of base vectors (change this!)
QUERY_SIZE = 50          # Number of queries (change this!)
DIMENSION = 128          # Vector dimension

# Parameter grid (adjust these!)
PARAM_GRID = {
    'n_clusters': [10, 25, 50],           # K-Means clusters
    'k_children': [200, 500, 1000],       # Children per cluster  
    'child_search_ef': [50, 100, 200]     # HNSW search width
}

EVAL_PARAMS = {
    'k_values': [10],                     # Recall@k
    'n_probe_values': [3, 5, 10]         # Clusters to probe
}

MAX_COMBINATIONS = 12    # Test 12 out of 27 combinations
USE_SIFT_DATA = False    # Set True to use SIFT, False for synthetic

# ===================================

def main():
    print("Custom K-Means HNSW Parameter Tuning")
    print(f"Dataset size: {DATASET_SIZE}, Query size: {QUERY_SIZE}")
    
    # Load or create data
    if USE_SIFT_DATA:
        base_vectors, query_vectors = load_sift_data()
        if base_vectors is not None:
            # Use subset of SIFT data
            base_vectors = base_vectors[:DATASET_SIZE]
            query_vectors = query_vectors[:QUERY_SIZE]
        else:
            print("SIFT data not found, using synthetic")
            USE_SIFT_DATA = False
    
    if not USE_SIFT_DATA:
        # Create synthetic data
        np.random.seed(42)
        base_vectors = np.random.randn(DATASET_SIZE, DIMENSION).astype(np.float32)
        query_vectors = np.random.randn(QUERY_SIZE, DIMENSION).astype(np.float32)
    
    query_ids = list(range(len(query_vectors)))
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    print(f"Using {'SIFT' if USE_SIFT_DATA else 'synthetic'} data: "
          f"{base_vectors.shape} base, {query_vectors.shape} queries")
    
    # Build base HNSW index
    print("Building base HNSW index...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 1000 == 0:
            print(f"  Inserted {i + 1}/{len(base_vectors)} vectors")
    
    # Initialize evaluator
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # Parameter sweep
    print(f"\nStarting parameter sweep with {MAX_COMBINATIONS} combinations...")
    sweep_results = evaluator.parameter_sweep(
        base_index,
        PARAM_GRID,
        EVAL_PARAMS,
        max_combinations=MAX_COMBINATIONS
    )
    
    # Find optimal parameters
    optimal = evaluator.find_optimal_parameters(
        sweep_results,
        optimization_target='recall_at_k',
        constraints={'avg_query_time_ms': 50.0}  # Max 50ms per query
    )
    
    if optimal:
        print(f"\nOptimal parameters found!")
        print(f"Parameters: {optimal['parameters']}")
        print(f"Recall@10: {optimal['performance']['recall_at_k']:.4f}")
        print(f"Query time: {optimal['performance']['avg_query_time_ms']:.2f}ms")
        
        # Build optimal system and compare with baseline
        print("\nBuilding optimal system...")
        optimal_system = KMeansHNSW(base_index=base_index, **optimal['parameters'])
        
        comparison = evaluator.compare_with_baselines(
            optimal_system, base_index, k=10, n_probe=10, ef_values=[50, 100, 200]
        )
        
        print("\nComparison with baseline HNSW:")
        kmeans_perf = comparison['kmeans_hnsw']
        print(f"K-Means HNSW: Recall={kmeans_perf['recall_at_k']:.4f}, "
              f"Time={kmeans_perf['avg_query_time_ms']:.2f}ms")
        
        for baseline in comparison['baseline_hnsw']:
            print(f"HNSW (ef={baseline['ef']}): "
                  f"Recall={baseline['recall_at_k']:.4f}, "
                  f"Time={baseline['avg_query_time_ms']:.2f}ms")
        
        # Save results
        results = {
            'config': {
                'dataset_size': DATASET_SIZE,
                'query_size': QUERY_SIZE,
                'dimension': DIMENSION,
                'use_sift': USE_SIFT_DATA,
                'param_grid': PARAM_GRID,
                'max_combinations': MAX_COMBINATIONS
            },
            'sweep_results': sweep_results,
            'optimal_parameters': optimal,
            'baseline_comparison': comparison
        }
        
        filename = f'custom_tuning_results_{DATASET_SIZE}_{QUERY_SIZE}.json'
        save_results(results, filename)
        print(f"\nResults saved to {filename}")
    
    print("\nTuning completed!")

if __name__ == "__main__":
    main()
