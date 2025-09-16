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
        """K-means++ initialization for better initial centroids."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        for c_id in range(1, self.n_clusters):
            # Calculate distances to nearest centroid for each point
            distances = np.array([
                min([np.linalg.norm(x - c)**2 for c in centroids[:c_id]])
                for x in X
            ])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[c_id] = X[j]
                    break
                    
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = [np.linalg.norm(X[i] - centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
            
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as the mean of assigned points."""
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]
                
        return centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
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
            'std_cluster_size': np.std(cluster_sizes)
        }
