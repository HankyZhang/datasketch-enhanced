#!/usr/bin/env python3
"""
Enhanced Level 2 Validation with Recall Rate Analysis
Tests recall performance across different algorithms and methods before full 1M run

Algorithms tested:
1. Standard HNSW (approximate hierarchical search            print(f"    ef={ef}: R@1={result['recall_at_1']:.3f}, R@10={result['recall_at_10']:.3f}, "
                  f"R@100={result['recall_at_100']:.3f}, Time={result['avg_query_time_ms']:.2f}ms")
            
            # Warning for suspiciously high recall
            if result['recall_at_10'] > 0.95 and ef < 50:
                print(f"    ⚠️  Suspiciously high recall with low ef={ef}!")Hybrid HNSW with method variants:
   - approx: Fast approximate parent-child mapping
   - brute: Exact brute force parent-child mapping
   - diversify: Assignment limiting for balanced coverage
   - repair: Minimum coverage enforcement
"""

import numpy as np
import struct
import time
import json
from typing import Dict, List, Tuple, Optional

def read_fvecs(filename: str, max_count: int = None) -> np.ndarray:
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

def read_ivecs(filename: str, max_count: int = None) -> np.ndarray:
    """Read .ivecs format files efficiently."""
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
                
            vector = struct.unpack('i' * dim, vector_bytes)
            vectors.append(vector)
            count += 1
    
    return np.array(vectors, dtype=np.int32)

def compute_ground_truth_subset(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute ground truth using brute force for subset testing."""
    print(f"Computing ground truth for {len(query_vectors)} queries...")
    
    ground_truth = []
    for i, query in enumerate(query_vectors):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(query_vectors)}")
            
        distances = np.linalg.norm(base_vectors - query, axis=1)
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        ground_truth.append(nearest_indices)
    
    return np.array(ground_truth)

def calculate_recall_at_k(search_results: List[int], ground_truth: List[int], k: int) -> float:
    """Calculate recall@k metric."""
    if not search_results or not ground_truth:
        return 0.0
    
    k_results = search_results[:k]
    k_truth = ground_truth[:k]
    
    intersection = len(set(k_results) & set(k_truth))
    return intersection / len(k_truth)

def evaluate_recall_performance(
    index, 
    query_vectors: np.ndarray, 
    ground_truth: np.ndarray,
    search_params: Dict,
    algorithm_name: str
) -> Dict:
    """Evaluate recall performance for any HNSW variant."""
    
    print(f"  Evaluating {algorithm_name} recall...")
    
    recalls_1 = []
    recalls_10 = []
    recalls_100 = []
    query_times = []
    
    for i, query in enumerate(query_vectors):
        start_time = time.time()
        
        # Handle different search interfaces
        if hasattr(index, 'search'):  # Hybrid HNSW
            results = index.search(query, k=100, **search_params)
            result_ids = [rid for rid, _ in results]
        else:  # Standard HNSW
            results = index.query(query, k=100, **search_params)
            result_ids = [rid for rid, _ in results]
        
        query_time = (time.time() - start_time) * 1000  # ms
        query_times.append(query_time)
        
        # Calculate recall at different k values
        gt_ids = ground_truth[i].tolist()
        
        recall_1 = calculate_recall_at_k(result_ids, gt_ids, 1)
        recall_10 = calculate_recall_at_k(result_ids, gt_ids, 10)
        recall_100 = calculate_recall_at_k(result_ids, gt_ids, 100)
        
        recalls_1.append(recall_1)
        recalls_10.append(recall_10)
        recalls_100.append(recall_100)
    
    return {
        'algorithm': algorithm_name,
        'search_params': search_params,
        'avg_query_time_ms': np.mean(query_times),
        'recall_at_1': np.mean(recalls_1),
        'recall_at_10': np.mean(recalls_10),
        'recall_at_100': np.mean(recalls_100),
        'std_query_time_ms': np.std(query_times),
        'std_recall_at_10': np.std(recalls_10)
    }

def load_ground_truth(query_count: int, base_count: int) -> np.ndarray:
    """Load ground truth - prefer real SIFT GT when available, fallback to computed."""
    try:
        # Try to load real SIFT ground truth
        print("🎯 Loading real SIFT ground truth...")
        real_gt = read_ivecs("sift/sift_groundtruth.ivecs", query_count)
        
        # Filter to only include indices that exist in our base subset
        filtered_gt = []
        valid_queries = []
        
        for i in range(len(real_gt)):
            valid_indices = [idx for idx in real_gt[i] if idx < base_count]
            if len(valid_indices) >= 10:  # Ensure we have enough valid ground truth
                # Pad or truncate to exactly 100 elements
                if len(valid_indices) >= 100:
                    filtered_gt.append(valid_indices[:100])
                else:
                    # Pad with -1 (invalid) if not enough valid indices
                    padded = valid_indices + [-1] * (100 - len(valid_indices))
                    filtered_gt.append(padded)
                valid_queries.append(i)
            else:
                print(f"  Warning: Query {i} has only {len(valid_indices)} valid GT indices")
                
        if len(filtered_gt) >= query_count // 2:  # If we have valid GT for at least half queries
            print(f"  Using real SIFT ground truth for {len(filtered_gt)} queries")
            print(f"  Falling back to computed GT for queries with insufficient real GT")
            # Return the filtered ground truth as a proper numpy array
            return np.array(filtered_gt), valid_queries
        else:
            print("  Insufficient valid real ground truth, computing subset GT...")
            
    except FileNotFoundError:
        print("  Real SIFT ground truth not found, computing subset GT...")
    except Exception as e:
        print(f"  Error loading real ground truth: {e}")
    
    # Fallback to computed ground truth
    return None, None

def comprehensive_recall_validation():
    """Comprehensive recall validation across algorithm variants."""
    print("🎯 COMPREHENSIVE RECALL VALIDATION")
    print("Testing Standard HNSW vs Hybrid HNSW variants with recall metrics")
    print("=" * 70)
    
    # Import modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Load test dataset - Use larger dataset and more challenging parameters
    print("📚 Loading test dataset...")
    base_vectors = read_fvecs("sift/sift_base.fvecs", 100000)  # 100K for realistic testing
    query_vectors = read_fvecs("sift/sift_query.fvecs", 100)   # 100 queries
    
    print(f"Dataset: {len(base_vectors)} base vectors, {len(query_vectors)} queries")
    
    # Compute ground truth for reliable testing
    print("🎯 Computing ground truth...")
    ground_truth = compute_ground_truth_subset(base_vectors, query_vectors, k=100)
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    all_results = []
    
    # 1. Standard HNSW Testing
    print(f"\n🔬 STANDARD HNSW TESTING")
    print("-" * 40)
    
    hnsw_configs = [
        {"m": 8, "ef_construction": 100, "name": "Low Quality"},      # Reduced quality
        {"m": 16, "ef_construction": 200, "name": "Standard"},
        {"m": 16, "ef_construction": 400, "name": "High Quality"},
        {"m": 24, "ef_construction": 400, "name": "High Connectivity"},
    ]
    
    for config in hnsw_configs:
        print(f"\n📋 Testing {config['name']} HNSW...")
        
        # Build index
        start_time = time.time()
        hnsw_index = HNSW(distance_func=distance_func, m=config['m'], ef_construction=config['ef_construction'])
        hnsw_index.update(dataset)
        build_time = time.time() - start_time
        
        print(f"  Built in {build_time:.2f}s")
        
        # Test different ef values - Include lower ef values to stress-test recall
        for ef in [10, 20, 50, 100, 200, 400]:  # Added lower ef values
            result = evaluate_recall_performance(
                hnsw_index, 
                query_vectors, 
                ground_truth,
                {'ef': ef},
                f"{config['name']} HNSW (ef={ef})"
            )
            result['build_time'] = build_time
            result['config'] = config
            all_results.append(result)
            
            print(f"    ef={ef}: R@1={result['recall_at_1']:.3f}, R@10={result['recall_at_10']:.3f}, "
                  f"R@100={result['recall_at_100']:.3f}, Time={result['avg_query_time_ms']:.2f}ms")
            
            # Warning for suspiciously high recall
            if result['recall_at_10'] > 0.95 and ef < 50:
                print(f"    ⚠️  Suspiciously high recall with low ef={ef}!")
    
    # 2. Hybrid HNSW Testing with Method Variants
    print(f"\n🚀 HYBRID HNSW TESTING - METHOD VARIANTS")
    print("-" * 50)
    
    # Build base index for hybrid testing
    base_hnsw = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    base_hnsw.update(dataset)
    
    hybrid_configs = [
        {
            "parent_level": 2, 
            "k_children": 1000, 
            "method": "approx", 
            "name": "Level 2 Approx",
            "diversify": None,
            "repair": None
        },
        {
            "parent_level": 2, 
            "k_children": 1000, 
            "method": "brute", 
            "name": "Level 2 Brute",
            "diversify": None,
            "repair": None
        },
        {
            "parent_level": 2, 
            "k_children": 1000, 
            "method": "approx", 
            "name": "Level 2 + Diversify",
            "diversify": 3,
            "repair": None
        },
        {
            "parent_level": 2, 
            "k_children": 1000, 
            "method": "approx", 
            "name": "Level 2 + Repair",
            "diversify": None,
            "repair": 1
        },
        {
            "parent_level": 2, 
            "k_children": 1000, 
            "method": "approx", 
            "name": "Level 2 + Both",
            "diversify": 3,
            "repair": 1
        },
        {
            "parent_level": 1, 
            "k_children": 1500, 
            "method": "approx", 
            "name": "Level 1 Baseline",
            "diversify": None,
            "repair": None
        },
    ]
    
    for config in hybrid_configs:
        print(f"\n📋 Testing {config['name']}...")
        
        try:
            # Build hybrid index
            start_time = time.time()
            hybrid_index = HNSWHybrid(
                base_index=base_hnsw,
                parent_level=config['parent_level'],
                k_children=config['k_children'],
                parent_child_method=config['method'],
                diversify_max_assignments=config['diversify'],
                repair_min_assignments=config['repair']
            )
            build_time = time.time() - start_time
            
            stats = hybrid_index.get_stats()
            num_parents = stats.get('num_parents', 0)
            print(f"  Built in {build_time:.2f}s, {num_parents} parents")
            
            # Test different n_probe values
            max_n_probe = max(1, int(0.5 * num_parents)) if config['parent_level'] == 2 else num_parents // 2
            n_probe_values = [min(n, max_n_probe) for n in [1, 2, 5, 10, 20] if min(n, max_n_probe) >= 1]
            
            for n_probe in n_probe_values:
                if n_probe > num_parents:
                    continue
                    
                result = evaluate_recall_performance(
                    hybrid_index,
                    query_vectors,
                    ground_truth,
                    {'n_probe': n_probe},
                    f"{config['name']} (n_probe={n_probe})"
                )
                result['build_time'] = build_time
                result['config'] = config
                result['stats'] = stats
                all_results.append(result)
                
                print(f"    n_probe={n_probe}: R@1={result['recall_at_1']:.3f}, R@10={result['recall_at_10']:.3f}, "
                      f"R@100={result['recall_at_100']:.3f}, Time={result['avg_query_time_ms']:.2f}ms")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    # 3. Analysis and Comparison
    print(f"\n📊 RECALL ANALYSIS & ALGORITHM COMPARISON")
    print("=" * 70)
    
    # Sort by recall@10
    all_results.sort(key=lambda x: x['recall_at_10'], reverse=True)
    
    print(f"\n🏆 TOP CONFIGURATIONS BY RECALL@10:")
    print(f"{'Rank':<4} {'Algorithm':<25} {'R@10':<8} {'R@100':<8} {'Time(ms)':<9} {'Method':<12}")
    print("-" * 75)
    
    for i, result in enumerate(all_results[:15], 1):
        method_info = ""
        if 'Hybrid' in result['algorithm']:
            config = result['config']
            method_parts = [config['method']]
            if config['diversify']:
                method_parts.append(f"div={config['diversify']}")
            if config['repair']:
                method_parts.append(f"rep={config['repair']}")
            method_info = ",".join(method_parts)
        else:
            method_info = "hierarchical"
            
        print(f"{i:<4} {result['algorithm'][:24]:<25} {result['recall_at_10']:<8.3f} "
              f"{result['recall_at_100']:<8.3f} {result['avg_query_time_ms']:<9.2f} {method_info:<12}")
    
    # Method comparison
    print(f"\n🔬 METHOD EFFECTIVENESS ANALYSIS:")
    print("-" * 40)
    
    standard_best = max([r for r in all_results if 'HNSW' in r['algorithm'] and 'Hybrid' not in r['algorithm']], 
                       key=lambda x: x['recall_at_10'], default=None)
    hybrid_best = max([r for r in all_results if 'Hybrid' in r['algorithm']], 
                     key=lambda x: x['recall_at_10'], default=None)
    
    if standard_best:
        print(f"🥇 Best Standard HNSW:")
        print(f"   Recall@10: {standard_best['recall_at_10']:.3f}")
        print(f"   Query time: {standard_best['avg_query_time_ms']:.2f}ms")
        print(f"   Method: Hierarchical navigation")
    
    if hybrid_best:
        print(f"🚀 Best Hybrid HNSW:")
        print(f"   Recall@10: {hybrid_best['recall_at_10']:.3f}")
        print(f"   Query time: {hybrid_best['avg_query_time_ms']:.2f}ms")
        config = hybrid_best['config']
        print(f"   Method: {config['method']} mapping, level {config['parent_level']}")
        if config['diversify']:
            print(f"   Features: diversify_max={config['diversify']}")
        if config['repair']:
            print(f"   Features: repair_min={config['repair']}")
    
    # Method analysis
    approx_results = [r for r in all_results if r['config'].get('method') == 'approx']
    brute_results = [r for r in all_results if r['config'].get('method') == 'brute']
    
    if approx_results and brute_results:
        avg_approx_recall = np.mean([r['recall_at_10'] for r in approx_results])
        avg_brute_recall = np.mean([r['recall_at_10'] for r in brute_results])
        avg_approx_time = np.mean([r['avg_query_time_ms'] for r in approx_results])
        avg_brute_time = np.mean([r['avg_query_time_ms'] for r in brute_results])
        
        print(f"\n⚖️  APPROX vs BRUTE METHOD COMPARISON:")
        print(f"   Approx: {avg_approx_recall:.3f} recall, {avg_approx_time:.2f}ms avg")
        print(f"   Brute:  {avg_brute_recall:.3f} recall, {avg_brute_time:.2f}ms avg")
        print(f"   Tradeoff: {'Brute' if avg_brute_recall > avg_approx_recall else 'Approx'} wins on recall")
    
    # Save results
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_info': {
            'base_vectors': len(base_vectors),
            'query_vectors': len(query_vectors),
            'algorithms_tested': len(set(r['algorithm'] for r in all_results))
        },
        'results': all_results,
        'best_standard_hnsw': standard_best,
        'best_hybrid_hnsw': hybrid_best,
        'method_analysis': {
            'approx_avg_recall': np.mean([r['recall_at_10'] for r in approx_results]) if approx_results else None,
            'brute_avg_recall': np.mean([r['recall_at_10'] for r in brute_results]) if brute_results else None,
        }
    }
    
    with open('recall_validation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: recall_validation_results.json")
    
    return all_results

def test_parameter_sensitivity():
    """Test how parameters affect recall performance and parent distribution."""
    print(f"\n🧪 PARAMETER SENSITIVITY ANALYSIS")
    print("How different parameters affect Level 2 recall and structure")
    print("=" * 70)
    
    # Import modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Load test dataset - Use larger dataset to avoid perfect recall
    base_vectors = read_fvecs("sift/sift_base.fvecs", 50000)  # 50K for sensitivity testing
    query_vectors = read_fvecs("sift/sift_query.fvecs", 50)   # 50 queries
    ground_truth = compute_ground_truth_subset(base_vectors, query_vectors, k=100)
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    print(f"\n📊 Dataset: {len(base_vectors)} vectors, {len(query_vectors)} queries")
    
    # Test different m values
    print(f"\n📈 EFFECT OF 'm' PARAMETER:")
    print(f"{'m':<4} {'Parents':<8} {'Max n_probe':<12} {'Recall@10':<10} {'Query(ms)':<10}")
    print("-" * 50)
    
    for m in [8, 16, 24, 32]:
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_index = HNSW(distance_func=distance_func, m=m, ef_construction=200)
        base_index.update(dataset)
        
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=1000, parent_child_method='approx')
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        max_n_probe = int(0.5 * num_parents) if num_parents > 0 else 0
        
        # Test recall with moderate n_probe
        test_n_probe = min(5, max_n_probe) if max_n_probe > 0 else 1
        result = evaluate_recall_performance(
            hybrid_index, query_vectors, ground_truth, 
            {'n_probe': test_n_probe}, f"m={m}"
        )
        
        print(f"{m:<4} {num_parents:<8} {max_n_probe:<12} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f}")
    
    # Test different ef_construction values
    print(f"\n📈 EFFECT OF 'ef_construction' PARAMETER:")
    print(f"{'ef_c':<6} {'Parents':<8} {'Max n_probe':<12} {'Recall@10':<10} {'Query(ms)':<10}")
    print("-" * 55)
    
    for ef_c in [100, 200, 400, 600]:
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_index = HNSW(distance_func=distance_func, m=16, ef_construction=ef_c)
        base_index.update(dataset)
        
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=1000, parent_child_method='approx')
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        max_n_probe = int(0.5 * num_parents) if num_parents > 0 else 0
        
        # Test recall with moderate n_probe
        test_n_probe = min(5, max_n_probe) if max_n_probe > 0 else 1
        result = evaluate_recall_performance(
            hybrid_index, query_vectors, ground_truth,
            {'n_probe': test_n_probe}, f"ef_c={ef_c}"
        )
        
        print(f"{ef_c:<6} {num_parents:<8} {max_n_probe:<12} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f}")
    
    # Test k_children effect on recall
    print(f"\n📈 EFFECT OF 'k_children' PARAMETER:")
    print(f"{'k_child':<8} {'Recall@10':<10} {'Query(ms)':<10} {'Coverage':<10}")
    print("-" * 45)
    
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    base_index.update(dataset)
    
    for k_children in [500, 1000, 1500, 2000]:
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=k_children, parent_child_method='approx')
        stats = hybrid_index.get_stats()
        coverage = stats.get('coverage_fraction', 0)
        
        # Test with fixed n_probe
        result = evaluate_recall_performance(
            hybrid_index, query_vectors, ground_truth,
            {'n_probe': 5}, f"k_children={k_children}"
        )
        
        print(f"{k_children:<8} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f} {coverage:<10.3f}")

def validate_1m_readiness():
    """Final validation check for 1M dataset readiness."""
    print(f"\n🔮 1M DATASET READINESS ASSESSMENT")
    print("=" * 50)
    
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Test with largest feasible subset
    print("📊 Testing scalability with 100K subset...")
    base_vectors = read_fvecs("sift/sift_base.fvecs", 100000)  # 100K
    query_vectors = read_fvecs("sift/sift_query.fvecs", 20)     # 20 queries
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    # Build base index
    print("🔧 Building base HNSW index...")
    start_time = time.time()
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    base_index.update(dataset)
    base_build_time = time.time() - start_time
    
    # Test Level 2 hybrid with constraint
    print("🚀 Building Level 2 hybrid index...")
    start_time = time.time()
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=2000,
        parent_child_method='approx'
    )
    hybrid_build_time = time.time() - start_time
    
    stats = hybrid_index.get_stats()
    num_parents = stats.get('num_parents', 0)
    max_n_probe = int(0.5 * num_parents)
    
    print(f"\n📈 100K Results:")
    print(f"  Base build time: {base_build_time:.2f}s")
    print(f"  Hybrid build time: {hybrid_build_time:.2f}s")
    print(f"  Parents at level 2: {num_parents:,}")
    print(f"  Max n_probe (constraint): {max_n_probe}")
    print(f"  Coverage: {stats.get('coverage_fraction', 0):.3f}")
    
    # Estimate 1M performance
    scale_factor = 1000000 / 100000  # 10x
    est_base_build_1m = base_build_time * scale_factor * 1.2  # 20% overhead
    est_hybrid_build_1m = hybrid_build_time * scale_factor * 1.1  # 10% overhead
    est_parents_1m = num_parents * scale_factor
    est_max_n_probe_1m = int(0.5 * est_parents_1m)
    
    print(f"\n🔮 1M Estimates:")
    print(f"  Estimated base build: ~{est_base_build_1m/60:.1f} minutes")
    print(f"  Estimated hybrid build: ~{est_hybrid_build_1m/60:.1f} minutes")
    print(f"  Estimated parents: ~{est_parents_1m:,.0f}")
    print(f"  Estimated max n_probe: ~{est_max_n_probe_1m:,}")
    
    # Readiness assessment
    readiness_score = 0
    issues = []
    
    if est_max_n_probe_1m >= 10:
        readiness_score += 25
        print("✅ n_probe range: Good (≥10)")
    else:
        issues.append("Limited n_probe range")
        print("⚠️  n_probe range: Limited (<10)")
    
    if est_base_build_1m < 1800:  # 30 minutes
        readiness_score += 25
        print("✅ Build time: Reasonable (<30min)")
    else:
        issues.append("Long build time")
        print("⚠️  Build time: Long (>30min)")
    
    if stats.get('coverage_fraction', 0) > 0.8:
        readiness_score += 25
        print("✅ Coverage: Good (>80%)")
    else:
        issues.append("Low coverage")
        print("⚠️  Coverage: Low (<80%)")
    
    if num_parents > 50:
        readiness_score += 25
        print("✅ Parent count: Sufficient (>50)")
    else:
        issues.append("Few parents")
        print("⚠️  Parent count: Low (<50)")
    
    print(f"\n🎯 READINESS SCORE: {readiness_score}/100")
    
    if readiness_score >= 75:
        print("🚀 READY FOR 1M TESTING!")
        recommendation = "Proceed with test_1m_sift_level2.py"
    elif readiness_score >= 50:
        print("⚠️  PROCEED WITH CAUTION")
        recommendation = "Consider parameter adjustments before 1M test"
    else:
        print("❌ NOT READY")
        recommendation = "Address issues before 1M test"
    
    if issues:
        print(f"⚠️  Issues: {', '.join(issues)}")
    
    print(f"💡 Recommendation: {recommendation}")
    
    return readiness_score >= 50

def main():
    """Enhanced main validation function with comprehensive recall testing."""
    print("🔬 ENHANCED LEVEL 2 VALIDATION WITH RECALL ANALYSIS")
    print("Testing recall performance across algorithm variants")
    print("=" * 70)
    
    try:
        # Check if SIFT data exists
        import os
        if not os.path.exists("sift/sift_base.fvecs"):
            print("❌ SIFT dataset not found!")
            print("Please ensure the sift/ directory exists with SIFT dataset files.")
            return
        
        # Run comprehensive recall validation
        results = comprehensive_recall_validation()
        
        # Run parameter sensitivity analysis
        test_parameter_sensitivity()
        
        # Validate 1M readiness
        ready = validate_1m_readiness()
        
        print(f"\n{'='*70}")
        print("📊 VALIDATION COMPLETE")
        print(f"{'='*70}")
        
        # Summary of best performers
        if results:
            best_overall = max(results, key=lambda x: x['recall_at_10'])
            print(f"🏆 Best Overall Recall@10: {best_overall['recall_at_10']:.3f}")
            print(f"   Algorithm: {best_overall['algorithm']}")
            
            # Recall distribution analysis
            all_recalls = [r['recall_at_10'] for r in results]
            perfect_recall_count = sum(1 for r in all_recalls if r >= 0.99)
            print(f"\n📈 Recall Distribution:")
            print(f"   Perfect recall (≥0.99): {perfect_recall_count}/{len(all_recalls)} configs")
            print(f"   Average recall@10: {np.mean(all_recalls):.3f}")
            print(f"   Recall range: {min(all_recalls):.3f} - {max(all_recalls):.3f}")
            
            if perfect_recall_count > len(all_recalls) * 0.8:
                print(f"   ⚠️  {perfect_recall_count/len(all_recalls)*100:.0f}% configs show near-perfect recall!")
                print(f"   This suggests dataset may be too small or parameters too high-quality.")
            
            # Method effectiveness
            hybrid_results = [r for r in results if 'Hybrid' in r['algorithm']]
            if hybrid_results:
                method_performance = {}
                for result in hybrid_results:
                    method = result['config'].get('method', 'unknown')
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(result['recall_at_10'])
                
                print(f"\n📈 Method Performance Summary:")
                for method, recalls in method_performance.items():
                    avg_recall = np.mean(recalls)
                    std_recall = np.std(recalls)
                    print(f"   {method.capitalize()}: {avg_recall:.3f} ± {std_recall:.3f} avg recall@10")
        
        if ready:
            print(f"\n🚀 System ready for 1M dataset testing!")
            print(f"💡 Next step: py test_1m_sift_level2.py")
        else:
            print(f"\n⚠️  Consider parameter tuning before 1M test")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
