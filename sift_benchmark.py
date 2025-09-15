#!/usr/bin/env python3
"""
SIFT Dataset Benchmark for HNSW Enhanced

This script provides comprehensive benchmarking of HNSW and Hybrid HNSW
using the standard SIFT1M dataset. It handles the dataset properly and
provides meaningful recall metrics.

Usage:
    python sift_benchmark.py --size small    # 5K vectors, fast test
    python sift_benchmark.py --size medium   # 50K vectors, balanced
    python sift_benchmark.py --size large    # 200K vectors, comprehensive
    python sift_benchmark.py --size full     # 1M vectors, full dataset
"""

import numpy as np
import struct
import time
import argparse
import os
from typing import Dict, List, Tuple, Optional
import json

def read_fvecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .fvecs format files efficiently."""
    vectors = []
    count = 0
    
    with open(filename, 'rb') as f:
        while True:
            if max_count and count >= max_count:
                break
                
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
                
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
            
            if count % 10000 == 0:
                print(f"  Loaded {count} vectors...")
    
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .ivecs format files efficiently."""
    vectors = []
    count = 0
    
    with open(filename, 'rb') as f:
        while True:
            if max_count and count >= max_count:
                break
                
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
                
            vector = struct.unpack('i' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
    
    return np.array(vectors, dtype=np.int32)

def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute ground truth using brute force (for subset testing)."""
    print(f"Computing ground truth for {len(query_vectors)} queries against {len(base_vectors)} base vectors...")
    
    ground_truth = []
    for i, query in enumerate(query_vectors):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(query_vectors)}")
            
        # Compute distances to all base vectors
        distances = np.linalg.norm(base_vectors - query, axis=1)
        
        # Get k nearest neighbors
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        
        ground_truth.append(nearest_indices)
    
    return np.array(ground_truth)

class SIFTBenchmark:
    """SIFT benchmark for HNSW systems."""
    
    def __init__(self, dataset_size: str = "small"):
        self.dataset_size = dataset_size
        self.size_configs = {
            "small": {"base": 5000, "queries": 100},
            "medium": {"base": 50000, "queries": 500}, 
            "large": {"base": 200000, "queries": 1000},
            "full": {"base": None, "queries": None}  # Load all
        }
        
    def load_data(self):
        """Load SIFT dataset according to size configuration."""
        config = self.size_configs[self.dataset_size]
        
        print(f"Loading SIFT dataset (size: {self.dataset_size})...")
        
        # Load base vectors
        print("Loading base vectors...")
        self.base_vectors = read_fvecs("sift/sift_base.fvecs", config["base"])
        print(f"Loaded {len(self.base_vectors)} base vectors")
        
        # Load query vectors  
        print("Loading query vectors...")
        self.query_vectors = read_fvecs("sift/sift_query.fvecs", config["queries"])
        print(f"Loaded {len(self.query_vectors)} query vectors")
        
        # For subset testing, compute our own ground truth
        if self.dataset_size != "full":
            print("Computing ground truth for subset...")
            self.ground_truth = compute_ground_truth(self.base_vectors, self.query_vectors, k=100)
        else:
            print("Loading provided ground truth...")
            self.ground_truth = read_ivecs("sift/sift_groundtruth.ivecs", config["queries"])
        
        print(f"Dataset loaded: {self.base_vectors.shape[0]} base, {self.query_vectors.shape[0]} queries")
        print()
        
    def benchmark_standard_hnsw(self, m: int = 16, ef_construction: int = 200) -> Dict:
        """Benchmark standard HNSW."""
        from hnsw import HNSW
        
        print("=== Standard HNSW Benchmark ===")
        print(f"Parameters: m={m}, ef_construction={ef_construction}")
        
        # Build index
        print("Building HNSW index...")
        start_time = time.time()
        
        distance_func = lambda x, y: np.linalg.norm(x - y)
        index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
        
        dataset = {i: self.base_vectors[i] for i in range(len(self.base_vectors))}
        index.update(dataset)
        
        build_time = time.time() - start_time
        print(f"Build time: {build_time:.2f}s")
        
        # Test different ef values
        ef_values = [50, 100, 200, 400]
        results = {}
        
        for ef in ef_values:
            print(f"Testing with ef={ef}...")
            query_times = []
            recalls_at_k = {1: [], 10: [], 100: []}
            
            for i, query in enumerate(self.query_vectors):
                # Search
                start_time = time.time()
                search_results = index.query(query, k=100, ef=ef)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Calculate recall
                result_ids = [rid for rid, _ in search_results]
                gt_ids = self.ground_truth[i]
                
                for k in [1, 10, 100]:
                    k_min = min(k, len(result_ids), len(gt_ids))
                    if k_min > 0:
                        recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                        recalls_at_k[k].append(recall_k)
            
            avg_query_time = np.mean(query_times) * 1000  # ms
            avg_recalls = {k: np.mean(recalls) if recalls else 0.0 for k, recalls in recalls_at_k.items()}
            
            results[ef] = {
                'query_time_ms': avg_query_time,
                'recalls': avg_recalls
            }
            
            print(f"  ef={ef}: Query time={avg_query_time:.2f}ms, Recall@10={avg_recalls[10]:.4f}")
        
        return {
            'build_time': build_time,
            'results_by_ef': results,
            'parameters': {'m': m, 'ef_construction': ef_construction}
        }
    
    def benchmark_hybrid_hnsw(self, m: int = 16, ef_construction: int = 200, 
                            k_children: int = 1000) -> Dict:
        """Benchmark hybrid HNSW."""
        from hnsw import HNSW
        from hybrid_hnsw import HNSWHybrid
        
        print("=== Hybrid HNSW Benchmark ===")
        print(f"Parameters: m={m}, ef_construction={ef_construction}, k_children={k_children}")
        
        # Build base index
        print("Building base HNSW index...")
        start_time = time.time()
        
        distance_func = lambda x, y: np.linalg.norm(x - y)
        base_index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
        
        dataset = {i: self.base_vectors[i] for i in range(len(self.base_vectors))}
        base_index.update(dataset)
        
        base_build_time = time.time() - start_time
        
        # Build hybrid structure
        print("Building hybrid structure...")
        start_time = time.time()
        
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=2,
            k_children=k_children,
            parent_child_method='approx'
        )
        
        hybrid_build_time = time.time() - start_time
        total_build_time = base_build_time + hybrid_build_time
        
        print(f"Base build time: {base_build_time:.2f}s")
        print(f"Hybrid build time: {hybrid_build_time:.2f}s")
        
        # Get stats
        stats = hybrid_index.get_stats()
        print(f"Hybrid stats: {stats.get('num_parents', 0)} parents, {stats.get('num_children', 0)} children")
        
        # Test different n_probe values
        n_probe_values = [1, 2, 5, 10, 20]
        results = {}
        
        for n_probe in n_probe_values:
            print(f"Testing with n_probe={n_probe}...")
            query_times = []
            recalls_at_k = {1: [], 10: [], 100: []}
            
            for i, query in enumerate(self.query_vectors):
                # Search
                start_time = time.time()
                search_results = hybrid_index.search(query, k=100, n_probe=n_probe)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Calculate recall
                result_ids = [rid for rid, _ in search_results]
                gt_ids = self.ground_truth[i]
                
                for k in [1, 10, 100]:
                    k_min = min(k, len(result_ids), len(gt_ids))
                    if k_min > 0:
                        recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                        recalls_at_k[k].append(recall_k)
            
            avg_query_time = np.mean(query_times) * 1000  # ms
            avg_recalls = {k: np.mean(recalls) if recalls else 0.0 for k, recalls in recalls_at_k.items()}
            
            results[n_probe] = {
                'query_time_ms': avg_query_time,
                'recalls': avg_recalls
            }
            
            print(f"  n_probe={n_probe}: Query time={avg_query_time:.2f}ms, Recall@10={avg_recalls[10]:.4f}")
        
        return {
            'base_build_time': base_build_time,
            'hybrid_build_time': hybrid_build_time,
            'total_build_time': total_build_time,
            'results_by_probe': results,
            'stats': stats,
            'parameters': {'m': m, 'ef_construction': ef_construction, 'k_children': k_children}
        }
    
    def run_benchmark(self):
        """Run complete benchmark."""
        print(f"🚀 SIFT Benchmark - {self.dataset_size.upper()} Dataset")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Run benchmarks
        standard_results = self.benchmark_standard_hnsw()
        hybrid_results = self.benchmark_hybrid_hnsw()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 50)
        
        # Best standard result
        best_ef = max(standard_results['results_by_ef'].keys(),
                     key=lambda ef: standard_results['results_by_ef'][ef]['recalls'][10])
        best_standard = standard_results['results_by_ef'][best_ef]
        
        # Best hybrid result
        best_probe = max(hybrid_results['results_by_probe'].keys(),
                        key=lambda p: hybrid_results['results_by_probe'][p]['recalls'][10])
        best_hybrid = hybrid_results['results_by_probe'][best_probe]
        
        print(f"Dataset: {len(self.base_vectors)} vectors, {self.base_vectors.shape[1]} dimensions")
        print()
        print(f"Standard HNSW (ef={best_ef}):")
        print(f"  Build time: {standard_results['build_time']:.2f}s")
        print(f"  Query time: {best_standard['query_time_ms']:.2f}ms")
        print(f"  Recall@1:  {best_standard['recalls'][1]:.4f}")
        print(f"  Recall@10: {best_standard['recalls'][10]:.4f}")
        print()
        print(f"Hybrid HNSW (n_probe={best_probe}):")
        print(f"  Build time: {hybrid_results['total_build_time']:.2f}s")
        print(f"  Query time: {best_hybrid['query_time_ms']:.2f}ms")
        print(f"  Recall@1:  {best_hybrid['recalls'][1]:.4f}")
        print(f"  Recall@10: {best_hybrid['recalls'][10]:.4f}")
        print()
        
        # Speed comparison
        speedup = best_standard['query_time_ms'] / best_hybrid['query_time_ms']
        print(f"🏃 Query speedup: {speedup:.2f}x (Hybrid vs Standard)")
        
        # Save results
        results = {
            'dataset_size': self.dataset_size,
            'dataset_info': {
                'base_vectors': len(self.base_vectors),
                'query_vectors': len(self.query_vectors),
                'dimensions': self.base_vectors.shape[1]
            },
            'standard_hnsw': standard_results,
            'hybrid_hnsw': hybrid_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = f'sift_benchmark_{self.dataset_size}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to {filename}")
        print("✅ Benchmark completed!")

def main():
    parser = argparse.ArgumentParser(description="SIFT Dataset Benchmark for HNSW Enhanced")
    parser.add_argument("--size", choices=["small", "medium", "large", "full"], 
                       default="small", help="Dataset size to use")
    
    args = parser.parse_args()
    
    # Check if SIFT data exists
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found!")
        print("Please ensure the SIFT dataset is available in the 'sift' directory:")
        print("  sift/sift_base.fvecs")
        print("  sift/sift_query.fvecs")
        print("  sift/sift_groundtruth.ivecs")
        return
    
    # Run benchmark
    benchmark = SIFTBenchmark(args.size)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
