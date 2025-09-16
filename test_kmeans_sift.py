"""
Test K-means clustering on the SIFT dataset.
This script demonstrates how to use the K-means implementation with SIFT data.
"""

import numpy as np
import time
import json
import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kmeans import KMeans, load_sift_data, evaluate_clustering, benchmark_kmeans, find_optimal_k


def test_kmeans_on_sift(
    data_subset: str = "learn",  # "base", "learn", or "query"
    max_samples: int = 5000,     # Limit samples for faster testing
    k_values: list = None,       # List of k values to test
    save_results: bool = True    # Save results to JSON
) -> Dict[str, Any]:
    """
    Test K-means clustering on SIFT dataset.
    
    Args:
        data_subset: Which SIFT subset to use ("base", "learn", or "query")
        max_samples: Maximum number of samples to use
        k_values: List of k values to test (default: [10, 20, 50, 100])
        save_results: Whether to save results to JSON file
        
    Returns:
        Dictionary containing all test results
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]
    
    print("=" * 80)
    print("K-MEANS CLUSTERING TEST ON SIFT DATASET")
    print("=" * 80)
    
    # Load SIFT data
    try:
        base_vectors, learn_vectors, query_vectors, groundtruth = load_sift_data()
    except Exception as e:
        print(f"Error loading SIFT data: {e}")
        print("Please ensure SIFT dataset files are in the 'sift' directory")
        return {}
    
    # Select data subset
    if data_subset == "base":
        X = base_vectors
        print(f"Using base vectors: {X.shape}")
    elif data_subset == "learn":
        X = learn_vectors
        print(f"Using learn vectors: {X.shape}")
    elif data_subset == "query":
        X = query_vectors
        print(f"Using query vectors: {X.shape}")
    else:
        raise ValueError(f"Unknown data subset: {data_subset}")
    
    # Limit samples if requested
    if max_samples and len(X) > max_samples:
        print(f"Limiting to {max_samples} samples (from {len(X)})")
        # Use random sampling to get diverse subset
        np.random.seed(42)
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    print()
    
    # Initialize results dictionary
    results = {
        "dataset_info": {
            "subset": data_subset,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "data_range": [float(X.min()), float(X.max())],
            "data_mean": float(X.mean()),
            "data_std": float(X.std())
        },
        "k_means_results": {},
        "optimal_k_analysis": {},
        "execution_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "k_values_tested": k_values
        }
    }
    
    # Test 1: Find optimal k using elbow method
    print("🔍 FINDING OPTIMAL K USING ELBOW METHOD")
    print("-" * 50)
    try:
        start_time = time.time()
        optimal_k_elbow = find_optimal_k(X, k_range=(2, min(20, len(k_values) + 5)), method='elbow')
        elbow_time = time.time() - start_time
        
        print(f"Optimal k (elbow method): {optimal_k_elbow}")
        print(f"Time taken: {elbow_time:.2f} seconds")
        
        results["optimal_k_analysis"]["elbow_method"] = {
            "optimal_k": optimal_k_elbow,
            "computation_time": elbow_time
        }
    except Exception as e:
        print(f"Error in elbow method: {e}")
        optimal_k_elbow = k_values[0]
    
    print()
    
    # Test 2: Benchmark different k values
    print("📊 BENCHMARKING DIFFERENT K VALUES")
    print("-" * 50)
    
    kmeans_params = {
        'max_iters': 300,
        'tol': 1e-4,
        'n_init': 3,  # Reduced for faster testing
        'init': 'k-means++',
        'random_state': 42,
        'verbose': True
    }
    
    benchmark_results = benchmark_kmeans(X, k_values, **kmeans_params)
    results["k_means_results"] = benchmark_results
    
    # Test 3: Detailed analysis of best performing k
    print("\n🎯 DETAILED ANALYSIS OF BEST K")
    print("-" * 50)
    
    # Find best k based on silhouette score
    best_k = None
    best_silhouette = -1
    
    for k, metrics in benchmark_results.items():
        silhouette = metrics.get('silhouette_score')
        if silhouette is not None and silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    
    if best_k is None:
        best_k = optimal_k_elbow
    
    print(f"Best performing k: {best_k}")
    print(f"Best silhouette score: {best_silhouette:.4f}")
    
    # Run detailed analysis on best k
    print(f"\nRunning detailed analysis with k={best_k}...")
    start_time = time.time()
    
    best_kmeans = KMeans(
        n_clusters=best_k,
        max_iters=500,
        tol=1e-5,
        n_init=5,
        init='k-means++',
        random_state=42,
        verbose=True
    )
    
    best_kmeans.fit(X)
    detailed_metrics = evaluate_clustering(best_kmeans, X)
    
    analysis_time = time.time() - start_time
    detailed_metrics['detailed_analysis_time'] = analysis_time
    
    results["best_k_analysis"] = {
        "k": best_k,
        "metrics": detailed_metrics,
        "cluster_info": best_kmeans.get_cluster_info()
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset: SIFT {data_subset} subset ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"Optimal k (elbow): {optimal_k_elbow}")
    print(f"Best k (silhouette): {best_k}")
    print(f"Best silhouette score: {best_silhouette:.4f}")
    print(f"Best inertia: {detailed_metrics['inertia']:.2f}")
    print(f"Total execution time: {sum(r.get('fit_time', 0) for r in benchmark_results.values()) + elbow_time + analysis_time:.2f} seconds")
    
    # Cluster size distribution
    cluster_info = best_kmeans.get_cluster_info()
    print(f"\nCluster size distribution (k={best_k}):")
    print(f"  Average: {cluster_info['avg_cluster_size']:.1f}")
    print(f"  Std dev: {cluster_info['std_cluster_size']:.1f}")
    print(f"  Range: {min(cluster_info['cluster_sizes'].values())} - {max(cluster_info['cluster_sizes'].values())}")
    
    # Save results
    if save_results:
        output_file = f"kmeans_sift_{data_subset}_results.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return results


def quick_test():
    """Quick test with small dataset for validation."""
    print("🚀 QUICK TEST MODE")
    print("=" * 50)
    
    try:
        # Load a small subset for quick testing
        base_vectors, learn_vectors, query_vectors, groundtruth = load_sift_data()
        
        # Use learn vectors (smaller dataset)
        X = learn_vectors[:1000]  # Just 1000 samples
        print(f"Quick test with {X.shape[0]} samples")
        
        # Test with k=10
        kmeans = KMeans(n_clusters=10, max_iters=100, verbose=True, random_state=42)
        
        start_time = time.time()
        kmeans.fit(X)
        fit_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_clustering(kmeans, X)
        
        print(f"\nQuick test results:")
        print(f"  Fit time: {fit_time:.2f} seconds")
        print(f"  Inertia: {metrics['inertia']:.2f}")
        print(f"  Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
        print(f"  Iterations: {kmeans.n_iter_}")
        
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False


def main():
    """Main function to run K-means tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test K-means clustering on SIFT dataset")
    parser.add_argument("--subset", choices=["base", "learn", "query"], default="learn",
                       help="SIFT dataset subset to use")
    parser.add_argument("--max-samples", type=int, default=5000,
                       help="Maximum number of samples to use")
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 20, 50, 100],
                       help="K values to test")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test mode")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test()
        return 0 if success else 1
    
    try:
        results = test_kmeans_on_sift(
            data_subset=args.subset,
            max_samples=args.max_samples,
            k_values=args.k_values,
            save_results=not args.no_save
        )
        
        if results:
            print("\n✅ Test completed successfully!")
            return 0
        else:
            print("\n❌ Test failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
