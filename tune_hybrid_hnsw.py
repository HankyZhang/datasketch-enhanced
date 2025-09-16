#!/usr/bin/env python3
"""
Parameter Tuning for Hybrid HNSW

This script helps find optimal parameters for hybrid HNSW to achieve
better recall while maintaining speed advantages.
"""

import numpy as np
import struct
import time
import json
from typing import Dict, List, Tuple, Optional

def read_fvecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .fvecs format files efficiently."""
    vectors = []
    count = 0
    
    with open(filename, 'rb') as f:
        while True:
            if max_count and count >= max_count:
                break
                
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            
            vector_bytes = f.read(4 * dim)
            if len(vector_bytes) < 4 * dim:
                break
                
            vector = struct.unpack('f' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
    
    return np.array(vectors, dtype=np.float32)

def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute ground truth using brute force."""
    print(f"Computing ground truth for {len(query_vectors)} queries...")
    
    ground_truth = []
    for i, query in enumerate(query_vectors):
        distances = np.linalg.norm(base_vectors - query, axis=1)
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        ground_truth.append(nearest_indices)
    
    return np.array(ground_truth)

class HybridHNSWTuner:
    """Parameter tuning for Hybrid HNSW."""
    
    def __init__(self, base_size: int = 100000, query_size: int = 100):
        self.base_size = base_size
        self.query_size = query_size
        self.load_data()
        
    def load_data(self):
        """Load SIFT data subset."""
        print("Loading SIFT data for tuning...")
        self.base_vectors = read_fvecs("sift/sift_base.fvecs", self.base_size)
        self.query_vectors = read_fvecs("sift/sift_query.fvecs", self.query_size)
        self.ground_truth = compute_ground_truth(self.base_vectors, self.query_vectors, k=100)
        print(f"Loaded {len(self.base_vectors)} base vectors, {len(self.query_vectors)} queries")
        
    def test_configuration(self, m: int, ef_construction: int, parent_level: int, 
                          k_children: int, test_probes: List[int], approx_ef: Optional[int] = None,
                          diversify_max: Optional[int] = None, repair_min: Optional[int] = None) -> Dict:
        """Test a specific hybrid configuration."""
        from hnsw import HNSW
        from hybrid_hnsw import HNSWHybrid
        
        print(f"\nTesting: m={m}, ef_c={ef_construction}, level={parent_level}, k_children={k_children}, approx_ef={approx_ef}, div_max={diversify_max}, repair_min={repair_min}")
        
        # Build base index
        distance_func = lambda x, y: np.linalg.norm(x - y)
        base_index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
        
        dataset = {i: self.base_vectors[i] for i in range(len(self.base_vectors))}
        
        start_time = time.time()
        base_index.update(dataset)
        base_build_time = time.time() - start_time
        
        # Build hybrid structure with custom approx_ef and optimization parameters
        start_time = time.time()
        hybrid_index = HNSWHybrid(
            base_index=base_index,
            parent_level=parent_level,
            k_children=k_children,
            parent_child_method='exact',  # Use exact for better recall
            approx_ef=approx_ef,  # Set large approx_ef for high recall
            diversify_max_assignments=diversify_max,  # Control point reuse
            repair_min_assignments=repair_min         # Ensure coverage
        )
        hybrid_build_time = time.time() - start_time
        
        stats = hybrid_index.get_stats()
        actual_approx_ef = stats.get('approx_ef', hybrid_index.approx_ef)
        print(f"  Built: {stats.get('num_parents', 0)} parents, {stats.get('num_children', 0)} children, approx_ef={actual_approx_ef}")
        
        # Test different n_probe values
        results = []
        for n_probe in test_probes:
            query_times = []
            recalls_at_10 = []
            
            for i, query in enumerate(self.query_vectors):
                start_time = time.time()
                search_results = hybrid_index.search(query, k=100, n_probe=n_probe)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                result_ids = [rid for rid, _ in search_results]
                gt_ids = self.ground_truth[i]
                
                recall_10 = len(set(result_ids[:10]) & set(gt_ids[:10])) / 10
                recalls_at_10.append(recall_10)
            
            avg_query_time = np.mean(query_times) * 1000
            avg_recall = np.mean(recalls_at_10)
            
            results.append({
                'n_probe': n_probe,
                'query_time_ms': avg_query_time,
                'recall_at_10': avg_recall
            })
            
            print(f"    n_probe={n_probe}: {avg_query_time:.2f}ms, recall@10={avg_recall:.4f}")
        
        return {
            'config': {
                'm': m,
                'ef_construction': ef_construction,
                'parent_level': parent_level,
                'k_children': k_children,
                'approx_ef': actual_approx_ef,
                'diversify_max': diversify_max,
                'repair_min': repair_min
            },
            'build_time': base_build_time + hybrid_build_time,
            'stats': stats,
            'results': results
        }
    
    def grid_search(self):
        """Perform grid search over key parameters."""
        print("🔍 Hybrid HNSW Parameter Grid Search")
        print("=" * 50)
        
        # Define parameter grid - Focus on level 2 with high recall and optimization configurations
        configs = [
            # (m, ef_construction, parent_level, k_children, approx_ef, diversify_max, repair_min)
            (16, 200, 2, 500, 2000, None, None),    # Baseline without optimization
            (16, 200, 2, 500, 2000, 3, 1),         # With diversify and repair
        ]
        
        test_probes = [1, 2, 5, 10]  # Reduced probe range for faster testing
        all_results = []
        
        for config in configs:
            m, ef_construction, parent_level, k_children, approx_ef, diversify_max, repair_min = config
            try:
                result = self.test_configuration(m, ef_construction, parent_level, 
                                               k_children, test_probes, approx_ef, 
                                               diversify_max, repair_min)
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        # Analyze results
        print("\n" + "=" * 50)
        print("📊 ANALYSIS")
        print("=" * 50)
        
        best_configs = []
        
        # Find best configurations for different recall targets - Focus on high recall
        recall_targets = [0.8, 0.9, 0.95, 0.99]  # Higher recall targets
        
        for target_recall in recall_targets:
            best_config = None
            best_speed = float('inf')
            
            for result in all_results:
                config = result['config']
                
                # Find best n_probe for this recall target
                for probe_result in result['results']:
                    if probe_result['recall_at_10'] >= target_recall:
                        if probe_result['query_time_ms'] < best_speed:
                            best_speed = probe_result['query_time_ms']
                            best_config = {
                                'config': config,
                                'n_probe': probe_result['n_probe'],
                                'query_time_ms': probe_result['query_time_ms'],
                                'recall_at_10': probe_result['recall_at_10'],
                                'build_time': result['build_time']
                            }
                        break
            
            if best_config:
                best_configs.append((target_recall, best_config))
                print(f"\nBest for Recall@10 >= {target_recall:.1f}:")
                print(f"  Config: {best_config['config']}")
                print(f"  n_probe: {best_config['n_probe']}")
                print(f"  Query time: {best_config['query_time_ms']:.2f}ms")
                print(f"  Recall@10: {best_config['recall_at_10']:.4f}")
                print(f"  Build time: {best_config['build_time']:.2f}s")
        
        # Save results
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'base_vectors': len(self.base_vectors),
                'query_vectors': len(self.query_vectors)
            },
            'all_results': all_results,
            'best_configs': {str(target): config for target, config in best_configs}
        }
        
        with open('hybrid_hnsw_tuning.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n📁 Detailed results saved to hybrid_hnsw_tuning.json")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS FOR HIGHEST RECALL")
        print("=" * 40)
        
        # Find absolute highest recall configuration
        highest_recall = 0
        highest_recall_config = None
        
        for result in all_results:
            for probe_result in result['results']:
                if probe_result['recall_at_10'] > highest_recall:
                    highest_recall = probe_result['recall_at_10']
                    highest_recall_config = {
                        'config': result['config'],
                        'n_probe': probe_result['n_probe'],
                        'query_time_ms': probe_result['query_time_ms'],
                        'recall_at_10': probe_result['recall_at_10'],
                        'build_time': result['build_time']
                    }
        
        if highest_recall_config:
            print(f"HIGHEST RECALL ACHIEVED: {highest_recall:.4f}")
            print(f"  Config: {highest_recall_config['config']}")
            print(f"  n_probe: {highest_recall_config['n_probe']}")
            print(f"  Query time: {highest_recall_config['query_time_ms']:.2f}ms")
            print(f"  Build time: {highest_recall_config['build_time']:.2f}s")
        
        print("\nFor very high recall (>95%):")
        if best_configs and any(target >= 0.95 for target, _ in best_configs):
            config = next(config for target, config in best_configs if target >= 0.95)
            print(f"  Use: {config['config']} with n_probe={config['n_probe']}")
        else:
            print("  Consider increasing ef_construction, m, or k_children further")
        
        print("For high recall (>90%):")
        if best_configs and any(target >= 0.9 for target, _ in best_configs):
            config = next(config for target, config in best_configs if target >= 0.9)
            print(f"  Use: {config['config']} with n_probe={config['n_probe']}")
        else:
            print("  Consider increasing k_children or ef_construction")
        
        print("For good recall (>80%):")
        if best_configs:
            config = best_configs[0][1]  # First available target
            print(f"  Use: {config['config']} with n_probe={config['n_probe']}")

def main():
    if not os.path.exists("sift"):
        print("❌ SIFT dataset not found! Please ensure sift/ directory exists.")
        return
    
    tuner = HybridHNSWTuner(base_size=10000, query_size=50)  # Smaller test for faster results
    tuner.grid_search()

if __name__ == "__main__":
    import os
    main()
