#!/usr/bin/env python3
"""
Comprehensive recall test for optimized Hybrid HNSW
"""

import sys
import os
sys.path.append('.')

import time
import numpy as np
from typing import Dict, List, Tuple
from datasketch import HNSW
from hnsw_hybrid_evaluation import generate_synthetic_dataset, create_query_set
from optimized_hybrid_hnsw import OptimizedHybridHNSW

def compute_ground_truth(dataset: Dict[int, np.ndarray], 
                        query_set: Dict[int, np.ndarray], 
                        k: int = 10) -> Dict[int, List[int]]:
    """Compute ground truth using brute force search"""
    print(f"Computing ground truth for {len(query_set)} queries...")
    ground_truth = {}
    
    for query_id, query_vector in query_set.items():
        distances = []
        for vec_id, vector in dataset.items():
            dist = np.linalg.norm(query_vector - vector)
            distances.append((vec_id, dist))
        
        distances.sort(key=lambda x: x[1])
        ground_truth[query_id] = [vec_id for vec_id, _ in distances[:k]]
    
    return ground_truth

def compute_recall(predicted: List[int], ground_truth: List[int]) -> float:
    """Compute recall@k"""
    if not ground_truth:
        return 0.0
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)

def test_hybrid_recall_comprehensive():
    """Comprehensive recall test for Hybrid HNSW"""
    print("🔍 COMPREHENSIVE HYBRID HNSW RECALL TEST")
    print("=" * 60)
    
    # Test parameters
    test_configs = [
        {"size": 1000, "dim": 32, "queries": 100, "k_children": 300, "n_probe": 5},
        {"size": 2000, "dim": 64, "queries": 200, "k_children": 500, "n_probe": 8},
        {"size": 5000, "dim": 64, "queries": 500, "k_children": 800, "n_probe": 12},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"📊 Testing: {config['size']} vectors, {config['dim']}D, {config['queries']} queries")
        print(f"   Parameters: k_children={config['k_children']}, n_probe={config['n_probe']}")
        print("="*60)
        
        # Generate dataset
        dataset = generate_synthetic_dataset(config['size'], config['dim'])
        query_set = create_query_set(dataset, config['queries'])
        
        print(f"✅ Generated {len(dataset)} vectors and {len(query_set)} queries")
        
        # Test 1: Baseline HNSW (reference)
        print("\n🔵 BASELINE HNSW (Reference)")
        print("-" * 30)
        
        baseline_start = time.time()
        baseline_hnsw = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
        baseline_hnsw.update(dataset)
        baseline_build_time = time.time() - baseline_start
        
        print(f"✅ Baseline build time: {baseline_build_time:.2f}s")
        
        # Test baseline recall
        baseline_search_times = []
        baseline_recalls = []
        
        for query_id, query_vector in list(query_set.items())[:50]:  # Test subset for speed
            start = time.time()
            baseline_results = baseline_hnsw.query(query_vector, k=10)
            search_time = time.time() - start
            baseline_search_times.append(search_time)
            
            # Get ground truth for this query
            distances = [(vid, np.linalg.norm(query_vector - vec)) for vid, vec in dataset.items()]
            distances.sort(key=lambda x: x[1])
            gt = [vid for vid, _ in distances[:10]]
            
            # Compute recall
            predicted = [vid for vid, _ in baseline_results]
            recall = compute_recall(predicted, gt)
            baseline_recalls.append(recall)
        
        baseline_avg_recall = np.mean(baseline_recalls)
        baseline_avg_search = np.mean(baseline_search_times) * 1000
        
        print(f"✅ Baseline recall@10: {baseline_avg_recall:.3f} ({baseline_avg_recall*100:.1f}%)")
        print(f"✅ Baseline search time: {baseline_avg_search:.2f}ms")
        
        # Test 2: Optimized Hybrid HNSW
        print("\n🟢 OPTIMIZED HYBRID HNSW")
        print("-" * 30)
        
        hybrid = OptimizedHybridHNSW(
            k_children=config['k_children'], 
            n_probe=config['n_probe']
        )
        
        # Build hybrid index
        hybrid_start = time.time()
        hybrid.build_base_index_optimized(dataset, m=8, ef_construction=50)
        hybrid.extract_parent_nodes(target_level=2)
        hybrid.build_parent_child_mapping_fast()
        hybrid_build_time = time.time() - hybrid_start
        
        print(f"✅ Hybrid build time: {hybrid_build_time:.2f}s")
        
        # Test hybrid recall  
        hybrid_search_times = []
        hybrid_recalls = []
        
        for query_id, query_vector in list(query_set.items())[:50]:  # Same subset
            start = time.time()
            hybrid_results = hybrid.search(query_vector, k=10)
            search_time = time.time() - start
            hybrid_search_times.append(search_time)
            
            # Get ground truth for this query
            distances = [(vid, np.linalg.norm(query_vector - vec)) for vid, vec in dataset.items()]
            distances.sort(key=lambda x: x[1])
            gt = [vid for vid, _ in distances[:10]]
            
            # Compute recall
            predicted = [vid for vid, _ in hybrid_results]
            recall = compute_recall(predicted, gt)
            hybrid_recalls.append(recall)
        
        hybrid_avg_recall = np.mean(hybrid_recalls)
        hybrid_avg_search = np.mean(hybrid_search_times) * 1000
        
        print(f"✅ Hybrid recall@10: {hybrid_avg_recall:.3f} ({hybrid_avg_recall*100:.1f}%)")
        print(f"✅ Hybrid search time: {hybrid_avg_search:.2f}ms")
        
        # Coverage analysis
        total_coverage = set()
        for children in hybrid.parent_child_map.values():
            total_coverage.update(children)
        coverage_pct = len(total_coverage) / len(dataset) * 100
        
        print(f"✅ Hybrid coverage: {coverage_pct:.1f}%")
        print(f"✅ Parent nodes: {len(hybrid.parent_ids)}")
        
        # Store results
        result = {
            'size': config['size'],
            'k_children': config['k_children'],
            'n_probe': config['n_probe'],
            'baseline_recall': baseline_avg_recall,
            'hybrid_recall': hybrid_avg_recall,
            'baseline_search_ms': baseline_avg_search,
            'hybrid_search_ms': hybrid_avg_search,
            'baseline_build_s': baseline_build_time,
            'hybrid_build_s': hybrid_build_time,
            'coverage_pct': coverage_pct,
            'parent_count': len(hybrid.parent_ids)
        }
        all_results.append(result)
        
        # Quick comparison
        recall_diff = hybrid_avg_recall - baseline_avg_recall
        speed_ratio = baseline_avg_search / hybrid_avg_search if hybrid_avg_search > 0 else 1
        
        print(f"\n📈 COMPARISON:")
        print(f"   Recall difference: {recall_diff:+.3f} ({recall_diff*100:+.1f} pp)")
        print(f"   Search speed ratio: {speed_ratio:.1f}x {'faster' if speed_ratio > 1 else 'slower'}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    print(f"{'Size':<6} {'Hybrid Recall':<13} {'Baseline Recall':<15} {'Coverage':<10} {'Parents':<8}")
    print("-" * 60)
    
    for result in all_results:
        print(f"{result['size']:<6} "
              f"{result['hybrid_recall']*100:<13.1f}% "
              f"{result['baseline_recall']*100:<15.1f}% "
              f"{result['coverage_pct']:<10.1f}% "
              f"{result['parent_count']:<8}")
    
    avg_hybrid_recall = np.mean([r['hybrid_recall'] for r in all_results])
    avg_baseline_recall = np.mean([r['baseline_recall'] for r in all_results])
    
    print(f"\nOverall average:")
    print(f"  Hybrid recall: {avg_hybrid_recall*100:.1f}%")
    print(f"  Baseline recall: {avg_baseline_recall*100:.1f}%")
    print(f"  Difference: {(avg_hybrid_recall - avg_baseline_recall)*100:+.1f} percentage points")
    
    return all_results

if __name__ == "__main__":
    results = test_hybrid_recall_comprehensive()
