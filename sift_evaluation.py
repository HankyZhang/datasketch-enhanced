#!/usr/bin/env python3
"""
SIFT Dataset Evaluation for HNSW and Hybrid HNSW

This script loads the SIFT1M dataset and evaluates both standard HNSW 
and hybrid HNSW implementations for performance comparison.

SIFT Dataset Structure:
- sift_base.fvecs: 1M base vectors (128-dim)
- sift_query.fvecs: 10K query vectors (128-dim) 
- sift_groundtruth.ivecs: Ground truth nearest neighbors
- sift_learn.fvecs: Learning vectors (not used here)
"""

import numpy as np
import struct
import time
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json

def read_fvecs(filename: str) -> np.ndarray:
    """
    Read .fvecs format files (float vectors).
    
    Format: [dim][vector1][dim][vector2]... where dim is int32 and vectors are float32
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
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
    
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename: str) -> np.ndarray:
    """
    Read .ivecs format files (integer vectors).
    
    Format: [dim][vector1][dim][vector2]... where dim is int32 and vectors are int32
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
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
    
    return np.array(vectors, dtype=np.int32)

class SIFTEvaluator:
    """Evaluator for SIFT dataset using HNSW and Hybrid HNSW."""
    
    def __init__(self, sift_dir: str = "sift"):
        """Initialize with SIFT dataset directory."""
        self.sift_dir = sift_dir
        self.base_vectors = None
        self.query_vectors = None
        self.ground_truth = None
        
    def load_dataset(self, max_base: Optional[int] = None, max_queries: Optional[int] = None):
        """
        Load SIFT dataset.
        
        Args:
            max_base: Maximum number of base vectors to load (for memory/speed)
            max_queries: Maximum number of queries to test
        """
        print("Loading SIFT dataset...")
        
        # Load base vectors
        print("Loading base vectors...")
        base_path = os.path.join(self.sift_dir, "sift_base.fvecs")
        self.base_vectors = read_fvecs(base_path)
        if max_base:
            self.base_vectors = self.base_vectors[:max_base]
        print(f"Loaded {len(self.base_vectors)} base vectors (128-dim)")
        
        # Load query vectors
        print("Loading query vectors...")
        query_path = os.path.join(self.sift_dir, "sift_query.fvecs")
        self.query_vectors = read_fvecs(query_path)
        if max_queries:
            self.query_vectors = self.query_vectors[:max_queries]
        print(f"Loaded {len(self.query_vectors)} query vectors")
        
        # Load ground truth
        print("Loading ground truth...")
        gt_path = os.path.join(self.sift_dir, "sift_groundtruth.ivecs")
        self.ground_truth = read_ivecs(gt_path)
        if max_queries:
            self.ground_truth = self.ground_truth[:max_queries]
        print(f"Loaded ground truth for {len(self.ground_truth)} queries")
        
        print(f"Dataset loaded successfully!")
        print(f"Base vectors shape: {self.base_vectors.shape}")
        print(f"Query vectors shape: {self.query_vectors.shape}")
        print(f"Ground truth shape: {self.ground_truth.shape}")
        print()
        
    def evaluate_standard_hnsw(self, m: int = 16, ef_construction: int = 200, ef_search: int = 200) -> Dict:
        """Evaluate standard HNSW performance."""
        from hnsw import HNSW
        
        print(f"=== Standard HNSW Evaluation ===")
        print(f"Parameters: m={m}, ef_construction={ef_construction}, ef_search={ef_search}")
        
        # Distance function (L2)
        distance_func = lambda x, y: np.linalg.norm(x - y)
        
        # Build index
        print("Building HNSW index...")
        start_time = time.time()
        index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
        
        # Insert vectors
        dataset = {i: self.base_vectors[i] for i in range(len(self.base_vectors))}
        index.update(dataset)
        
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.2f} seconds")
        
        # Evaluate queries
        print("Evaluating queries...")
        query_times = []
        recalls_at_k = {1: [], 10: [], 100: []}
        
        for i, query in enumerate(self.query_vectors):
            if i % 100 == 0:
                print(f"  Query {i+1}/{len(self.query_vectors)}")
                
            # Search
            start_time = time.time()
            results = index.query(query, k=100, ef=ef_search)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Calculate recall
            result_ids = [rid for rid, _ in results]
            gt_ids = self.ground_truth[i]
            
            for k in [1, 10, 100]:
                recall_k = len(set(result_ids[:k]) & set(gt_ids[:k])) / k
                recalls_at_k[k].append(recall_k)
        
        # Aggregate results
        avg_query_time = np.mean(query_times)
        avg_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
        
        results = {
            'type': 'standard_hnsw',
            'build_time': build_time,
            'avg_query_time_ms': avg_query_time * 1000,
            'recalls': avg_recalls,
            'parameters': {'m': m, 'ef_construction': ef_construction, 'ef_search': ef_search}
        }
        
        print(f"Results:")
        print(f"  Build time: {build_time:.2f}s")
        print(f"  Avg query time: {avg_query_time*1000:.2f}ms")
        print(f"  Recall@1: {avg_recalls[1]:.4f}")
        print(f"  Recall@10: {avg_recalls[10]:.4f}")
        print(f"  Recall@100: {avg_recalls[100]:.4f}")
        print()
        
        return results
    
    def evaluate_hybrid_hnsw(self, m: int = 16, ef_construction: int = 200, 
                           parent_level: int = 2, k_children: int = 1000) -> Dict:
        """Evaluate hybrid HNSW performance."""
        from hnsw import HNSW
        from hybrid_hnsw import HNSWHybrid
        
        print(f"=== Hybrid HNSW Evaluation ===")
        print(f"Parameters: m={m}, ef_construction={ef_construction}")
        print(f"Hybrid: parent_level={parent_level}, k_children={k_children}")
        
        # Distance function (L2)
        distance_func = lambda x, y: np.linalg.norm(x - y)
        
        # Build base index
        print("Building base HNSW index...")
        start_time = time.time()
        base_index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
        
        # Insert vectors
        dataset = {i: self.base_vectors[i] for i in range(len(self.base_vectors))}
        base_index.update(dataset)
        base_build_time = time.time() - start_time
        
        # Build hybrid index
        print("Building hybrid structure...")
        start_time = time.time()
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=parent_level,
            k_children=k_children,
            parent_child_method='approx'
        )
        hybrid_build_time = time.time() - start_time
        total_build_time = base_build_time + hybrid_build_time
        
        print(f"Base index built in {base_build_time:.2f}s")
        print(f"Hybrid structure built in {hybrid_build_time:.2f}s")
        print(f"Total build time: {total_build_time:.2f}s")
        
        # Get hybrid stats
        stats = hybrid_index.get_stats()
        print(f"Hybrid stats: {stats.get('num_parents', 0)} parents, {stats.get('num_children', 0)} children")
        
        # Evaluate queries with different n_probe values
        n_probe_values = [1, 5, 10, 20, 50]
        results_by_probe = {}
        
        for n_probe in n_probe_values:
            print(f"Evaluating with n_probe={n_probe}...")
            query_times = []
            recalls_at_k = {1: [], 10: [], 100: []}
            
            for i, query in enumerate(self.query_vectors):
                if i % 100 == 0:
                    print(f"  Query {i+1}/{len(self.query_vectors)}")
                    
                # Search
                start_time = time.time()
                results = hybrid_index.search(query, k=100, n_probe=n_probe)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Calculate recall
                result_ids = [rid for rid, _ in results]
                gt_ids = self.ground_truth[i]
                
                for k in [1, 10, 100]:
                    recall_k = len(set(result_ids[:k]) & set(gt_ids[:k])) / k
                    recalls_at_k[k].append(recall_k)
            
            # Store results for this n_probe
            avg_query_time = np.mean(query_times)
            avg_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
            
            results_by_probe[n_probe] = {
                'avg_query_time_ms': avg_query_time * 1000,
                'recalls': avg_recalls
            }
            
            print(f"  n_probe={n_probe}: Recall@10={avg_recalls[10]:.4f}, Time={avg_query_time*1000:.2f}ms")
        
        # Overall results
        results = {
            'type': 'hybrid_hnsw',
            'base_build_time': base_build_time,
            'hybrid_build_time': hybrid_build_time,
            'total_build_time': total_build_time,
            'results_by_probe': results_by_probe,
            'stats': stats,
            'parameters': {
                'm': m, 
                'ef_construction': ef_construction,
                'parent_level': parent_level,
                'k_children': k_children
            }
        }
        
        print()
        return results
    
    def compare_performance(self, save_results: bool = True):
        """Compare standard vs hybrid HNSW performance."""
        print("🚀 SIFT Dataset Performance Comparison")
        print("=" * 50)
        
        # Evaluate both approaches
        standard_results = self.evaluate_standard_hnsw()
        hybrid_results = self.evaluate_hybrid_hnsw()
        
        # Print comparison
        print("=== Performance Comparison ===")
        print(f"Standard HNSW:")
        print(f"  Build time: {standard_results['build_time']:.2f}s")
        print(f"  Query time: {standard_results['avg_query_time_ms']:.2f}ms")
        print(f"  Recall@10: {standard_results['recalls'][10]:.4f}")
        
        print(f"\nHybrid HNSW (best n_probe):")
        best_probe = max(hybrid_results['results_by_probe'].keys(), 
                        key=lambda p: hybrid_results['results_by_probe'][p]['recalls'][10])
        best_hybrid = hybrid_results['results_by_probe'][best_probe]
        
        print(f"  Build time: {hybrid_results['total_build_time']:.2f}s")
        print(f"  Query time: {best_hybrid['avg_query_time_ms']:.2f}ms (n_probe={best_probe})")
        print(f"  Recall@10: {best_hybrid['recalls'][10]:.4f}")
        
        # Speed comparison
        speedup = standard_results['avg_query_time_ms'] / best_hybrid['avg_query_time_ms']
        print(f"\n🏃 Query speedup: {speedup:.2f}x")
        
        # Save results
        if save_results:
            results = {
                'dataset_info': {
                    'base_vectors': len(self.base_vectors),
                    'query_vectors': len(self.query_vectors),
                    'dimensions': self.base_vectors.shape[1]
                },
                'standard_hnsw': standard_results,
                'hybrid_hnsw': hybrid_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('sift_evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📊 Results saved to sift_evaluation_results.json")
        
        return standard_results, hybrid_results
    
    def plot_recall_vs_speed(self, hybrid_results: Dict):
        """Plot recall vs query speed trade-off for hybrid HNSW."""
        try:
            import matplotlib.pyplot as plt
            
            n_probes = []
            recalls = []
            query_times = []
            
            for n_probe, results in hybrid_results['results_by_probe'].items():
                n_probes.append(n_probe)
                recalls.append(results['recalls'][10])  # Recall@10
                query_times.append(results['avg_query_time_ms'])
            
            plt.figure(figsize=(10, 6))
            
            # Plot recall vs n_probe
            plt.subplot(1, 2, 1)
            plt.plot(n_probes, recalls, 'bo-')
            plt.xlabel('n_probe')
            plt.ylabel('Recall@10')
            plt.title('Recall@10 vs n_probe')
            plt.grid(True)
            
            # Plot query time vs n_probe
            plt.subplot(1, 2, 2)
            plt.plot(n_probes, query_times, 'ro-')
            plt.xlabel('n_probe')
            plt.ylabel('Query Time (ms)')
            plt.title('Query Time vs n_probe')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('sift_hybrid_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📈 Analysis plots saved to sift_hybrid_analysis.png")
            
        except ImportError:
            print("⚠️  matplotlib not available, skipping plots")

def main():
    """Main evaluation function."""
    print("SIFT Dataset Evaluation for HNSW Enhanced")
    print("=" * 50)
    
    # Check if SIFT data exists
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found in 'sift' directory")
        print("Please ensure the SIFT dataset files are available:")
        print("  - sift_base.fvecs")
        print("  - sift_query.fvecs") 
        print("  - sift_groundtruth.ivecs")
        return
    
    # Create evaluator
    evaluator = SIFTEvaluator()
    
    # Load dataset (use subset for faster testing)
    print("Select dataset size:")
    print("1. Small test (10K vectors, 100 queries)")
    print("2. Medium test (100K vectors, 1K queries)")
    print("3. Full dataset (1M vectors, 10K queries)")
    
    choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        evaluator.load_dataset(max_base=10000, max_queries=100)
    elif choice == "2":
        evaluator.load_dataset(max_base=100000, max_queries=1000)
    else:
        evaluator.load_dataset()  # Full dataset
    
    # Run comparison
    standard_results, hybrid_results = evaluator.compare_performance()
    
    # Generate plots
    evaluator.plot_recall_vs_speed(hybrid_results)
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()
