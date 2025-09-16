"""
Utility functions for k-means clustering and SIFT data handling.
"""

import numpy as np
import struct
import os
from typing import Tuple, Optional, Dict, Any
import time


def load_sift_data(data_dir: str = "sift") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SIFT dataset files.
    
    Args:
        data_dir: Directory containing SIFT files
        
    Returns:
        Tuple of (base_vectors, learn_vectors, query_vectors, groundtruth)
    """
    def read_fvecs(filename: str) -> np.ndarray:
        """Read .fvecs file format."""
        with open(filename, 'rb') as f:
            vectors = []
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) != dim * 4:
                    break
                vector = struct.unpack(f'{dim}f', vector_bytes)
                vectors.append(vector)
            
            return np.array(vectors, dtype=np.float32)
    
    def read_ivecs(filename: str) -> np.ndarray:
        """Read .ivecs file format."""
        with open(filename, 'rb') as f:
            vectors = []
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) != dim * 4:
                    break
                vector = struct.unpack(f'{dim}i', vector_bytes)
                vectors.append(vector)
            
            return np.array(vectors, dtype=np.int32)
    
    base_file = os.path.join(data_dir, "sift_base.fvecs")
    learn_file = os.path.join(data_dir, "sift_learn.fvecs")
    query_file = os.path.join(data_dir, "sift_query.fvecs")
    gt_file = os.path.join(data_dir, "sift_groundtruth.ivecs")
    
    print("Loading SIFT dataset...")
    start_time = time.time()
    
    base_vectors = read_fvecs(base_file)
    print(f"Loaded base vectors: {base_vectors.shape}")
    
    learn_vectors = read_fvecs(learn_file)
    print(f"Loaded learn vectors: {learn_vectors.shape}")
    
    query_vectors = read_fvecs(query_file)
    print(f"Loaded query vectors: {query_vectors.shape}")
    
    groundtruth = read_ivecs(gt_file)
    print(f"Loaded groundtruth: {groundtruth.shape}")
    
    load_time = time.time() - start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    
    return base_vectors, learn_vectors, query_vectors, groundtruth


def evaluate_clustering(kmeans_model, X: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Evaluate clustering results.
    
    Args:
        kmeans_model: Fitted K-means model
        X: Input data
        true_labels: Optional true labels for external validation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import (
        silhouette_score, 
        calinski_harabasz_score, 
        davies_bouldin_score,
        adjusted_rand_score,
        normalized_mutual_info_score
    )
    
    predicted_labels = kmeans_model.labels_
    n_samples, n_features = X.shape
    
    # Internal validation metrics
    metrics = {
        'inertia': kmeans_model.inertia_,
        'n_iterations': kmeans_model.n_iter_,
        'n_clusters': kmeans_model.n_clusters,
        'n_samples': n_samples,
        'n_features': n_features
    }
    
    try:
        # Silhouette score (higher is better)
        metrics['silhouette_score'] = silhouette_score(X, predicted_labels)
    except:
        metrics['silhouette_score'] = None
    
    try:
        # Calinski-Harabasz score (higher is better)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, predicted_labels)
    except:
        metrics['calinski_harabasz_score'] = None
    
    try:
        # Davies-Bouldin score (lower is better)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, predicted_labels)
    except:
        metrics['davies_bouldin_score'] = None
    
    # External validation metrics (if true labels provided)
    if true_labels is not None:
        try:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
        except:
            metrics['adjusted_rand_score'] = None
            
        try:
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
        except:
            metrics['normalized_mutual_info'] = None
    
    # Cluster distribution analysis
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
    metrics['avg_cluster_size'] = float(np.mean(counts))
    metrics['std_cluster_size'] = float(np.std(counts))
    metrics['min_cluster_size'] = int(np.min(counts))
    metrics['max_cluster_size'] = int(np.max(counts))
    
    return metrics


def benchmark_kmeans(X: np.ndarray, k_values: list, **kmeans_kwargs) -> Dict[int, Dict[str, Any]]:
    """
    Benchmark K-means with different k values.
    
    Args:
        X: Input data
        k_values: List of k values to test
        **kmeans_kwargs: Additional arguments for KMeans
        
    Returns:
        Dictionary with results for each k value
    """
    from .kmeans import KMeans
    
    results = {}
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        start_time = time.time()
        
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        
        fit_time = time.time() - start_time
        
        # Evaluate clustering
        metrics = evaluate_clustering(kmeans, X)
        metrics['fit_time'] = fit_time
        
        results[k] = metrics
        
        print(f"k={k}: Inertia={metrics['inertia']:.2f}, "
              f"Silhouette={metrics.get('silhouette_score', 'N/A')}, "
              f"Time={fit_time:.2f}s")
    
    return results


def find_optimal_k(X: np.ndarray, k_range: Tuple[int, int] = (2, 20), method: str = 'elbow') -> int:
    """
    Find optimal number of clusters using elbow method or silhouette analysis.
    
    Args:
        X: Input data
        k_range: Range of k values to test (min_k, max_k)
        method: Method to use ('elbow' or 'silhouette')
        
    Returns:
        Optimal k value
    """
    from .kmeans import KMeans
    
    min_k, max_k = k_range
    k_values = list(range(min_k, max_k + 1))
    
    if method == 'elbow':
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, verbose=False)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point using the "kneedle" method approximation
        # Calculate the rate of decrease
        diffs = np.diff(inertias)
        diff_diffs = np.diff(diffs)
        
        # Find the point where the rate of decrease starts to slow down significantly
        optimal_idx = np.argmax(diff_diffs) + 2  # +2 because we took two diffs
        optimal_k = k_values[min(optimal_idx, len(k_values) - 1)]
        
        return optimal_k
    
    elif method == 'silhouette':
        silhouette_scores = []
        for k in k_values:
            if k == 1:
                continue  # Silhouette score is not defined for k=1
            kmeans = KMeans(n_clusters=k, random_state=42, verbose=False)
            kmeans.fit(X)
            
            try:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append((k, score))
            except:
                continue
        
        if not silhouette_scores:
            return k_values[len(k_values) // 2]  # Return middle value if silhouette fails
        
        # Return k with highest silhouette score
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        return optimal_k
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'elbow' or 'silhouette'")


def create_sample_dataset(n_samples: int = 1000, n_features: int = 128, n_clusters: int = 10, 
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a sample dataset for testing K-means.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_clusters: Number of true clusters
        random_state: Random seed
        
    Returns:
        Tuple of (data, true_labels)
    """
    np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.uniform(-10, 10, (n_clusters, n_features))
    
    # Generate data points around centers
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    X = []
    y = []
    
    for i in range(n_clusters):
        # Add extra samples to first clusters if remainder exists
        cluster_size = samples_per_cluster + (1 if i < remainder else 0)
        
        # Generate points around this center
        cluster_data = np.random.normal(centers[i], 2.0, (cluster_size, n_features))
        cluster_labels = np.full(cluster_size, i)
        
        X.append(cluster_data)
        y.append(cluster_labels)
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X.astype(np.float32), y.astype(np.int32)
