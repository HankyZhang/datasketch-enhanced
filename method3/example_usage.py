"""
Method 3 Example Usage: K-Means HNSW System

This script demonstrates how to use the K-Means HNSW system for various scenarios,
including SIFT dataset evaluation and synthetic data testing.
"""

import os
import sys
import time
import numpy as np
from typing import Tuple, List

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from method3.tune_kmeans_hnsw import KMeansHNSWEvaluator, save_results, load_sift_data
from hnsw.hnsw import HNSW


def demo_synthetic_data():
    """Demonstrate K-Means HNSW on synthetic data."""
    print("=" * 60)
    print("DEMO 1: Synthetic Data Example")
    print("=" * 60)
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    n_vectors = 5000
    dim = 128
    n_queries = 50
    
    np.random.seed(42)
    dataset = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Create queries (random subset)
    query_indices = np.random.choice(n_vectors, size=n_queries, replace=False)
    query_vectors = dataset[query_indices]
    query_ids = query_indices.tolist()
    
    print(f"Dataset: {dataset.shape}, Queries: {query_vectors.shape}")
    
    # Build base HNSW index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(dataset):
        if i not in query_ids:  # Exclude queries from index
            base_index.insert(i, vector)
    
    print(f"Base index built with {len(base_index)} vectors")
    
    # Build K-Means HNSW system
    print("Building K-Means HNSW system...")
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=25,  # 25 clusters for 5K dataset
        k_children=500,
        kmeans_params={'verbose': False}  # Reduce verbosity for demo
    )
    
    print("System built successfully!")
    print(f"Stats: {kmeans_hnsw.get_stats()}")
    
    # Test search performance
    print("\nTesting search performance...")
    test_query = query_vectors[0]
    
    # Test different n_probe values
    for n_probe in [3, 5, 10]:
        start_time = time.time()
        results = kmeans_hnsw.search(test_query, k=10, n_probe=n_probe)
        search_time = (time.time() - start_time) * 1000
        
        print(f"n_probe={n_probe}: {len(results)} results in {search_time:.2f}ms")
        for i, (node_id, distance) in enumerate(results[:3]):
            print(f"  {i+1}. Node {node_id}: {distance:.4f}")
    
    return kmeans_hnsw, base_index, query_vectors, query_ids, dataset


def demo_sift_evaluation():
    """Demonstrate K-Means HNSW on SIFT dataset."""
    print("\n" + "=" * 60)
    print("DEMO 2: SIFT Dataset Evaluation")
    print("=" * 60)
    
    # Load SIFT data
    base_vectors, query_vectors = load_sift_data()
    if base_vectors is None:
        print("SIFT data not available, skipping SIFT demo.")
        return None
    
    # Use subset for demo (first 10K base vectors, 50 queries)
    base_vectors = base_vectors[:10000]
    query_vectors = query_vectors[:50]
    query_ids = list(range(len(query_vectors)))
    
    print(f"Using SIFT subset: {base_vectors.shape[0]} base, {query_vectors.shape[0]} queries")
    
    # Build base HNSW index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 1000 == 0:
            print(f"  Inserted {i + 1}/{len(base_vectors)} vectors")
    
    # Build K-Means HNSW system optimized for SIFT
    print("Building K-Means HNSW system for SIFT...")
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=50,  # 50 clusters for 10K SIFT vectors
        k_children=1000,
        child_search_ef=200,
        kmeans_params={
            'max_iters': 100,  # Faster convergence for demo
            'n_init': 5,
            'verbose': False
        }
    )
    
    # Evaluate recall performance
    print("Evaluating recall performance...")
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    results = evaluator.evaluate_recall(
        kmeans_hnsw,
        k=10,
        n_probe=10,
        exclude_query_ids=False  # Since queries are not in base index
    )
    
    print(f"\nSIFT Evaluation Results:")
    print(f"  Recall@10: {results['recall_at_k']:.4f}")
    print(f"  Avg query time: {results['avg_query_time_ms']:.2f}ms")
    print(f"  Total evaluation time: {results['total_evaluation_time']:.2f}s")
    
    # Compare with baseline HNSW
    print("\nComparing with baseline HNSW...")
    comparison = evaluator.compare_with_baselines(
        kmeans_hnsw,
        base_index,
        k=10,
        n_probe=10,
        ef_values=[50, 100, 200]
    )
    
    print("Comparison Results:")
    kmeans_recall = comparison['kmeans_hnsw']['recall_at_k']
    kmeans_time = comparison['kmeans_hnsw']['avg_query_time_ms']
    print(f"  K-Means HNSW: Recall={kmeans_recall:.4f}, Time={kmeans_time:.2f}ms")
    
    for baseline in comparison['baseline_hnsw']:
        print(f"  Baseline (ef={baseline['ef']}): "
              f"Recall={baseline['recall_at_k']:.4f}, "
              f"Time={baseline['avg_query_time_ms']:.2f}ms")
    
    return results, comparison


def demo_parameter_optimization():
    """Demonstrate parameter optimization for K-Means HNSW."""
    print("\n" + "=" * 60)
    print("DEMO 3: Parameter Optimization")
    print("=" * 60)
    
    # Create small dataset for quick optimization demo
    print("Creating optimization dataset...")
    np.random.seed(123)
    dataset = np.random.randn(2000, 64).astype(np.float32)
    query_vectors = np.random.randn(20, 64).astype(np.float32)
    query_ids = list(range(len(query_vectors)))
    
    # Build base index
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(dataset):
        base_index.insert(i, vector)
    
    print(f"Built base index with {len(base_index)} vectors")
    
    # Setup evaluator
    evaluator = KMeansHNSWEvaluator(dataset, query_vectors, query_ids, distance_func)
    
    # Define parameter grid (small for demo)
    param_grid = {
        'n_clusters': [10, 20],
        'k_children': [200, 400],
        'child_search_ef': [50, 100]
    }
    
    evaluation_params = {
        'k_values': [10],
        'n_probe_values': [5, 10]
    }
    
    print("Running parameter optimization...")
    sweep_results = evaluator.parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=8  # All combinations for small grid
    )
    
    # Find optimal parameters
    optimal = evaluator.find_optimal_parameters(
        sweep_results,
        optimization_target='recall_at_k',
        constraints={'avg_query_time_ms': 50.0}
    )
    
    if optimal:
        print(f"\nOptimal parameters found:")
        print(f"  Parameters: {optimal['parameters']}")
        print(f"  Recall@10: {optimal['performance']['recall_at_k']:.4f}")
        print(f"  Query time: {optimal['performance']['avg_query_time_ms']:.2f}ms")
        print(f"  Construction time: {optimal['construction_time']:.2f}s")
    
    return sweep_results, optimal


def demo_advanced_features():
    """Demonstrate advanced features of K-Means HNSW."""
    print("\n" + "=" * 60)
    print("DEMO 4: Advanced Features")
    print("=" * 60)
    
    # Create dataset with clusters for interesting K-Means behavior
    print("Creating clustered dataset...")
    np.random.seed(456)
    
    # Create 5 distinct clusters
    n_clusters = 5
    points_per_cluster = 400
    dim = 32
    
    dataset = []
    cluster_centers = np.random.randn(n_clusters, dim) * 5  # Spread out centers
    
    for i in range(n_clusters):
        cluster_points = np.random.randn(points_per_cluster, dim) * 0.5 + cluster_centers[i]
        dataset.append(cluster_points)
    
    dataset = np.vstack(dataset).astype(np.float32)
    query_vectors = np.random.randn(10, dim).astype(np.float32)
    query_ids = list(range(len(query_vectors)))
    
    print(f"Created dataset with {len(dataset)} vectors in {n_clusters} clusters")
    
    # Build base index
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    for i, vector in enumerate(dataset):
        base_index.insert(i, vector)
    
    # Build K-Means HNSW with advanced features
    print("Building K-Means HNSW with advanced features...")
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=n_clusters,  # Match true number of clusters
        k_children=300,
        include_centroids_in_results=True,  # Include centroids
        diversify_max_assignments=3,  # Limit assignments per child
        repair_min_assignments=1,  # Ensure minimum coverage
        kmeans_params={
            'init': 'k-means++',
            'n_init': 10,
            'verbose': True
        }
    )
    
    # Show detailed statistics
    stats = kmeans_hnsw.get_stats()
    print(f"\nDetailed Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test search with different parameters
    print("\nTesting search with different parameters...")
    test_query = query_vectors[0]
    
    for n_probe in [1, 3, 5]:
        results = kmeans_hnsw.search(test_query, k=5, n_probe=n_probe)
        print(f"\nn_probe={n_probe} results:")
        for i, (node_id, distance) in enumerate(results):
            node_type = "centroid" if isinstance(node_id, str) and "centroid" in node_id else "data"
            print(f"  {i+1}. {node_type} {node_id}: {distance:.4f}")
    
    # Show centroid information
    centroid_info = kmeans_hnsw.get_centroid_info()
    print(f"\nCentroid Information:")
    print(f"  Number of centroids: {centroid_info['num_clusters']}")
    print(f"  Children per centroid (avg): {centroid_info['num_children'] / centroid_info['num_clusters']:.1f}")
    
    return kmeans_hnsw


def main():
    """Run all demonstration examples."""
    print("K-Means HNSW System - Comprehensive Demo")
    print("This demo showcases Method 3: K-Means-based Two-Stage HNSW")
    
    # Demo 1: Basic synthetic data usage
    synthetic_result = demo_synthetic_data()
    
    # Demo 2: SIFT dataset evaluation
    sift_result = demo_sift_evaluation()
    
    # Demo 3: Parameter optimization
    optimization_result = demo_parameter_optimization()
    
    # Demo 4: Advanced features
    advanced_result = demo_advanced_features()
    
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print("✓ Synthetic data demo completed")
    print("✓ SIFT evaluation completed" if sift_result else "⚠ SIFT data not available")
    print("✓ Parameter optimization completed")
    print("✓ Advanced features demo completed")
    print("\nAll demos completed successfully!")
    print("Method 3 (K-Means HNSW) is ready for use.")


if __name__ == "__main__":
    main()
