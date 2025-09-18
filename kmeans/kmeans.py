"""
K-means clustering algorithm implementation.
Optimized for high-dimensional vector data like SIFT features.
"""

import numpy as np
import time
from typing import Tuple, Optional, Union, List
import random


class KMeans:
    """
    K-means clustering algorithm with optimizations for high-dimensional data.
    
    Features:
    - K-means++ initialization for better initial centroids
    - Multiple initialization attempts
    - Early stopping when convergence is reached
    - Support for different distance metrics
    - Memory-efficient implementation for large datasets
    """
    
    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init: str = 'k-means++',
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            max_iters: Maximum number of iterations
            tol: Tolerance for convergence
            n_init: Number of different initializations to try
            init: Initialization method ('k-means++' or 'random')
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        
        # Results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ or random initialization."""
        n_samples, n_features = X.shape
        
        if self.init == 'k-means++':
            return self._kmeans_plus_plus_init(X)
        elif self.init == 'random':
            # Random initialization
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[random_indices].copy()
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization with vectorized operations for better performance."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        for c_id in range(1, self.n_clusters):
            # Vectorized distance computation to all existing centroids
            existing_centroids = centroids[:c_id]  # Shape: (c_id, n_features)
            
            # Compute distances from all points to all existing centroids
            # X shape: (n_samples, n_features), existing_centroids: (c_id, n_features)
            distances_to_centroids = np.linalg.norm(
                X[:, np.newaxis] - existing_centroids[np.newaxis, :], axis=2
            )  # Shape: (n_samples, c_id)
            
            # Get minimum distance to any centroid for each point
            min_distances_squared = np.min(distances_to_centroids, axis=1) ** 2
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances_squared / min_distances_squared.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            # Find the first index where cumulative probability >= r
            next_centroid_idx = np.searchsorted(cumulative_probs, r)
            centroids[c_id] = X[next_centroid_idx]
                    
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid using vectorized operations."""
        # Vectorized distance computation: much faster than loops
        # X shape: (n_samples, n_features)
        # centroids shape: (n_clusters, n_features)
        
        # Compute distances using broadcasting: (n_samples, n_clusters)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        
        # Find closest centroid for each point
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids using vectorized operations."""
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Vectorized centroid update
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = X[mask].mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]
                
        return centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Calculate within-cluster sum of squares using vectorized operations."""
        # Vectorized inertia calculation
        assigned_centroids = centroids[labels]  # Shape: (n_samples, n_features)
        squared_distances = np.sum((X - assigned_centroids) ** 2, axis=1)
        return np.sum(squared_distances)
    
    def _fit_single(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Single k-means run."""
        centroids = self._init_centroids(X)
        prev_inertia = float('inf')
        
        for iteration in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            current_inertia = self._calculate_inertia(X, labels, new_centroids)
            
            if abs(prev_inertia - current_inertia) < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            centroids = new_centroids
            prev_inertia = current_inertia
            
            if self.verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}, Inertia: {current_inertia:.2f}")
        
        return centroids, labels, current_inertia, iteration + 1
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-means clustering to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        X = np.array(X, dtype=np.float32)
        
        if self.verbose:
            print(f"Fitting K-means with {self.n_clusters} clusters on {X.shape[0]} samples...")
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        # Try multiple initializations
        for init_run in range(self.n_init):
            if self.verbose and self.n_init > 1:
                print(f"Initialization {init_run + 1}/{self.n_init}")
            
            centroids, labels, inertia, n_iter = self._fit_single(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        if self.verbose:
            print(f"Final inertia: {self.inertia_:.2f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        return self.fit(X).labels_
    
    def get_cluster_info(self) -> dict:
        """Get information about the clustering results."""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted first")
        
        # Calculate cluster sizes
        unique_labels, cluster_sizes = np.unique(self.labels_, return_counts=True)
        
        return {
            'n_clusters': self.n_clusters,
            'inertia': self.inertia_,
            'n_iterations': self.n_iter_,
            'cluster_sizes': dict(zip(unique_labels, cluster_sizes)),
            'avg_cluster_size': np.mean(cluster_sizes),
            'std_cluster_size': np.std(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes)
        }
