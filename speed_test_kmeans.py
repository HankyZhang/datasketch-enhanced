#!/usr/bin/env python3
"""
Speed test for optimized K-means implementation
"""

import sys
import time
import numpy as np
sys.path.append('.')

from sklearn.cluster import MiniBatchKMeans

def speed_test_kmeans():
    """Test K-means speed with different dataset sizes."""
    print("🚀 MiniBatchKMeans Speed Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {'n_samples': 1000, 'n_features': 128, 'n_clusters': 10},
        {'n_samples': 5000, 'n_features': 128, 'n_clusters': 20},
        {'n_samples': 10000, 'n_features': 128, 'n_clusters': 50},
    ]
    
    for config in test_configs:
        print(f"\nTest: {config['n_samples']} samples, {config['n_features']} features, {config['n_clusters']} clusters")
        print("-" * 50)

        # Create synthetic dataset
        np.random.seed(42)
        X = np.random.randn(config['n_samples'], config['n_features']).astype(np.float32)

        # Test MiniBatchKMeans
        print("Testing MiniBatchKMeans...")
        start_time = time.time()

        kmeans = MiniBatchKMeans(
            n_clusters=config['n_clusters'],
            max_iter=100,
            n_init=3,
            batch_size=min(1024, config['n_samples']),
            verbose=0,
            random_state=42
        )

        kmeans.fit(X)

        elapsed_time = time.time() - start_time

        print(f"✅ Completed in {elapsed_time:.2f} seconds")
        print(f"   Final inertia: {kmeans.inertia_:.2f}")
        print(f"   Iterations: {getattr(kmeans, 'n_iter_', 'N/A')}")

        # Performance metrics
        samples_per_second = config['n_samples'] / elapsed_time
        print(f"   Performance: {samples_per_second:.0f} samples/second")

def test_kmeans_hnsw_speed():
    """Test the full K-means HNSW system speed."""
    print("\n" + "=" * 60)
    print("🏗️  MiniBatchKMeans + HNSW System Speed Test")
    print("=" * 60)
    
    from hnsw.hnsw import HNSW
    from method3.kmeans_hnsw import KMeansHNSW
    
    # Create dataset
    n_vectors = 5000
    dim = 128
    np.random.seed(42)
    dataset = np.random.randn(n_vectors, dim).astype(np.float32)
    
    print(f"Dataset: {n_vectors} vectors, {dim} dimensions")
    
    # Build base HNSW
    print("Building base HNSW...")
    start_time = time.time()
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=100)
    
    for i, vector in enumerate(dataset):
        base_index.insert(i, vector)
    
    hnsw_time = time.time() - start_time
    print(f"HNSW built in {hnsw_time:.2f}s")
    
    # Build K-means HNSW system
    print("Building MiniBatchKMeans HNSW system...")
    start_time = time.time()
    
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=25,
        k_children=200,
        kmeans_params={'verbose': True}
    )
    
    total_time = time.time() - start_time
    print(f"K-means HNSW system built in {total_time:.2f}s")
    
    # Show breakdown
    stats = kmeans_hnsw.get_stats()
    print(f"\nTime breakdown:")
    print(f"  MiniBatchKMeans clustering: {stats['kmeans_fit_time']:.2f}s")
    print(f"  Child assignment: {stats['child_mapping_time']:.2f}s")
    print(f"  Total construction: {stats['total_construction_time']:.2f}s")
    
    # Test search speed
    print(f"\nTesting search speed...")
    query = np.random.randn(dim).astype(np.float32)
    
    search_times = []
    for _ in range(10):
        start = time.time()
        results = kmeans_hnsw.search(query, k=10, n_probe=5)
        search_times.append((time.time() - start) * 1000)
    
    avg_search_time = np.mean(search_times)
    print(f"Average search time: {avg_search_time:.2f}ms")
    print(f"Search results: {len(results)} neighbors found")

if __name__ == "__main__":
    speed_test_kmeans()
    test_kmeans_hnsw_speed()
    
    print("\n🎉 Speed test completed!")
    print("\nNotes:")
    print("✅ Using sklearn MiniBatchKMeans backend (stable & battle-tested)")
    print("✅ Vectorized distance & minibatch updates")
    print("✅ Adaptive batch size (≤1024)")
    print("✅ Compatible with KMeansHNSW clustering parameters")
