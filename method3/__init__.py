"""
Method 3: K-Means-based Two-Stage HNSW System

This package implements a two-stage retrieval system using K-Means clustering 
for parent discovery and HNSW for child assignment.

Main Components:
- KMeansHNSW: Core K-Means HNSW system
- KMeansHNSWEvaluator: Performance evaluation framework
- Parameter tuning and optimization tools
"""

from .kmeans_hnsw import KMeansHNSW
from .tune_kmeans_hnsw import KMeansHNSWEvaluator
from .kmeans_hnsw_multi_pivot import KMeansHNSWMultiPivot

__all__ = ['KMeansHNSW', 'KMeansHNSWEvaluator', 'KMeansHNSWMultiPivot']

# Version information
__version__ = '1.0.0'
__method__ = "K-Means HNSW"

# Import core classes when implemented
# from .method3_core import Method3Index
