"""
K-means clustering implementation for vector data.
"""

from .kmeans import KMeans
from .utils import load_sift_data, evaluate_clustering, benchmark_kmeans, find_optimal_k, create_sample_dataset

__version__ = "1.0.0"
__all__ = ["KMeans", "load_sift_data", "evaluate_clustering", "benchmark_kmeans", "find_optimal_k", "create_sample_dataset"]
