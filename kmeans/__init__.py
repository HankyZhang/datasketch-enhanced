"""
K-means clustering implementation for vector data.
"""

from .kmeans import KMeans
from .utils import load_sift_data, evaluate_clustering

__version__ = "1.0.0"
__all__ = ["KMeans", "load_sift_data", "evaluate_clustering"]
