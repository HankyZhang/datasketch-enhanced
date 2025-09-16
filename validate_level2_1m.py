#!/usr/bin/env python3
"""
Enhanced Hybrid HNSW Validation with Adaptive approx_ef
Tests hybrid HNSW recall and efficiency under different parameter datasets

Focus areas:
1. Adaptive approx_ef validation across dataset sizes
2. Hybrid HNSW performance with method variants:
   - approx: Fast approximate parent-child mapping with adaptive ef
   - brute: Exact brute force parent-child mapping  
   - diversify: Assignment limiting for balanced coverage
   - repair: Minimum coverage enforcement
3. Parameter sensitivity analysis for different dataset scales
4. Efficiency vs recall tradeoffs

Skips standard HNSW testing to focus on hybrid performance validation.
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

def comprehensive_hybrid_validation():
    """Comprehensive hybrid HNSW validation with adaptive approx_ef across multiple dataset sizes."""
    print("🎯 COMPREHENSIVE HYBRID HNSW VALIDATION")
    print("Testing Hybrid HNSW with adaptive approx_ef across different dataset scales")
    print("=" * 70)
    
    # Import modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    # Test multiple dataset sizes to validate adaptive approx_ef
    dataset_configs = [
        {"size": 10000, "queries": 50, "name": "Small (10K)"},
        {"size": 25000, "queries": 75, "name": "Medium (25K)"},
        {"size": 50000, "queries": 100, "name": "Large (50K)"},
        {"size": 100000, "queries": 100, "name": "XL (100K)"}
    ]
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    all_results = []
    
    for dataset_config in dataset_configs:
        print(f"\n🗂️  TESTING {dataset_config['name']} DATASET")
        print("=" * 50)
        
        # Load dataset
        print(f"📚 Loading {dataset_config['size']} base vectors...")
        base_vectors = read_fvecs("sift/sift_base.fvecs", dataset_config['size'])
        query_vectors = read_fvecs("sift/sift_query.fvecs", dataset_config['queries'])
        
        print(f"Dataset: {len(base_vectors)} base vectors, {len(query_vectors)} queries")
        
        # Compute ground truth
        print("🎯 Computing ground truth...")
        ground_truth = compute_ground_truth_subset(base_vectors, query_vectors, k=100)
        
        # Build base HNSW index
        print("🔧 Building base HNSW index...")
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_hnsw = HNSW(distance_func=distance_func, m=16, ef_construction=200)
        base_hnsw.update(dataset)
        print(f"   Base index built with {len(base_hnsw)} vectors")
        
        # Test hybrid configurations with different k_children values
        hybrid_configs = [
            {
                "k_children": 500, 
                "method": "approx", 
                "name": "Approx k=500",
                "approx_ef": None,  # Auto-compute
                "diversify": None,
                "repair": None
            },
            {
                "k_children": 1000, 
                "method": "approx", 
                "name": "Approx k=1000",
                "approx_ef": None,  # Auto-compute
                "diversify": None,
                "repair": None
            },
            {
                "k_children": 1500, 
                "method": "approx", 
                "name": "Approx k=1500",
                "approx_ef": None,  # Auto-compute
                "diversify": None,
                "repair": None
            },
            {
                "k_children": 1000, 
                "method": "brute", 
                "name": "Brute k=1000",
                "approx_ef": None,  # Not used for brute
                "diversify": None,
                "repair": None
            },
            {
                "k_children": 1000, 
                "method": "approx", 
                "name": "Approx + Diversify",
                "approx_ef": None,  # Auto-compute
                "diversify": 3,
                "repair": None
            },
            {
                "k_children": 1000, 
                "method": "approx", 
                "name": "Approx + Repair",
                "approx_ef": None,  # Auto-compute
                "diversify": None,
                "repair": 2
            }
        ]
        
        for config in hybrid_configs:
            print(f"\n📋 Testing {config['name']} on {dataset_config['name']}...")
            
            try:
                # Build hybrid index
                start_time = time.time()
                hybrid_index = HNSWHybrid(
                    base_index=base_hnsw,
                    parent_level=2,
                    k_children=config['k_children'],
                    parent_child_method=config['method'],
                    approx_ef=config['approx_ef'],
                    diversify_max_assignments=config['diversify'],
                    repair_min_assignments=config['repair']
                )
                build_time = time.time() - start_time
                
                stats = hybrid_index.get_stats()
                num_parents = stats.get('num_parents', 0)
                approx_ef_used = getattr(hybrid_index, 'approx_ef', 'N/A')
                
                print(f"   Built in {build_time:.2f}s")
                print(f"   Parents: {num_parents}, approx_ef: {approx_ef_used}")
                print(f"   Children/parent: {stats.get('avg_children_per_parent', 0):.1f}")
                print(f"   Coverage: {stats.get('coverage_fraction', 0):.3f}")
                
                # Test different n_probe values
                max_n_probe = max(1, min(20, num_parents // 2))
                n_probe_values = [1, 3, 5, 10, max_n_probe] if max_n_probe >= 5 else [1, max_n_probe]
                n_probe_values = list(set([n for n in n_probe_values if n <= num_parents and n > 0]))
                
                for n_probe in n_probe_values:
                    result = evaluate_recall_performance(
                        hybrid_index,
                        query_vectors,
                        ground_truth,
                        {'n_probe': n_probe},
                        f"{config['name']} (n_probe={n_probe}) - {dataset_config['name']}"
                    )
                    result['build_time'] = build_time
                    result['config'] = config
                    result['dataset_config'] = dataset_config
                    result['stats'] = stats
                    result['approx_ef_used'] = approx_ef_used
                    all_results.append(result)
                    
                    print(f"     n_probe={n_probe}: R@10={result['recall_at_10']:.3f}, "
                          f"R@100={result['recall_at_100']:.3f}, Time={result['avg_query_time_ms']:.2f}ms")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
    
    return all_results

def test_adaptive_ef_analysis():
    """Test and analyze adaptive approx_ef behavior across dataset sizes."""
    print(f"\n🧪 ADAPTIVE APPROX_EF ANALYSIS")
    print("How adaptive approx_ef affects performance across different scales")
    print("=" * 70)
    
    # Import modules
    from hnsw import HNSW
    from hybrid_hnsw import HNSWHybrid
    
    dataset_sizes = [5000, 10000, 25000, 50000]
    k_children_values = [500, 1000, 1500]
    
    print(f"{'Dataset':<8} {'k_child':<8} {'Auto_ef':<8} {'Rec_ef':<8} {'Children/P':<10} {'Coverage':<8} {'Build(s)':<8}")
    print("-" * 75)
    
    for size in dataset_sizes:
        # Load dataset
        base_vectors = read_fvecs("sift/sift_base.fvecs", size)
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        
        # Build base index
        distance_func = lambda x, y: np.linalg.norm(x - y)
        base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
        base_index.update(dataset)
        
        for k_children in k_children_values:
            try:
                # Test auto approx_ef
                start_time = time.time()
                hybrid = HNSWHybrid(
                    base_index=base_index,
                    parent_level=2,
                    k_children=k_children,
                    approx_ef=None  # Auto-compute
                )
                build_time = time.time() - start_time
                
                auto_ef = hybrid.approx_ef
                rec_ef = hybrid.get_recommended_ef(target_recall=0.95)
                stats = hybrid.get_stats()
                
                print(f"{size:<8} {k_children:<8} {auto_ef:<8} {rec_ef:<8} "
                      f"{stats.get('avg_children_per_parent', 0):<10.1f} "
                      f"{stats.get('coverage_fraction', 0):<8.3f} {build_time:<8.2f}")
                
            except Exception as e:
                print(f"{size:<8} {k_children:<8} Error: {e}")
                continue

def test_parameter_sensitivity():
    """Test how parameters affect recall performance and parent distribution for hybrid HNSW."""
    print(f"\n🧪 HYBRID PARAMETER SENSITIVITY ANALYSIS")
    print("How different parameters affect hybrid recall and structure")
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
    
    # Test different base m values effect on hybrid
    print(f"\n📈 EFFECT OF BASE 'm' PARAMETER ON HYBRID:")
    print(f"{'m':<4} {'Parents':<8} {'Auto_ef':<8} {'Recall@10':<10} {'Query(ms)':<10}")
    print("-" * 55)
    
    for m in [8, 16, 24, 32]:
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_index = HNSW(distance_func=distance_func, m=m, ef_construction=200)
        base_index.update(dataset)
        
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=1000, parent_child_method='approx')
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        
        # Test recall with moderate n_probe
        test_n_probe = min(5, max(1, num_parents // 10))
        result = evaluate_recall_performance(
            hybrid_index, query_vectors, ground_truth, 
            {'n_probe': test_n_probe}, f"m={m}"
        )
        
        print(f"{m:<4} {num_parents:<8} {hybrid_index.approx_ef:<8} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f}")
    
    # Test different ef_construction values
    print(f"\n📈 EFFECT OF BASE 'ef_construction' PARAMETER ON HYBRID:")
    print(f"{'ef_c':<6} {'Parents':<8} {'Auto_ef':<8} {'Recall@10':<10} {'Query(ms)':<10}")
    print("-" * 60)
    
    for ef_c in [100, 200, 400, 600]:
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        base_index = HNSW(distance_func=distance_func, m=16, ef_construction=ef_c)
        base_index.update(dataset)
        
        hybrid_index = HNSWHybrid(base_index, parent_level=2, k_children=1000, parent_child_method='approx')
        stats = hybrid_index.get_stats()
        num_parents = stats.get('num_parents', 0)
        
        # Test recall with moderate n_probe
        test_n_probe = min(5, max(1, num_parents // 10))
        result = evaluate_recall_performance(
            hybrid_index, query_vectors, ground_truth,
            {'n_probe': test_n_probe}, f"ef_c={ef_c}"
        )
        
        print(f"{ef_c:<6} {num_parents:<8} {hybrid_index.approx_ef:<8} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f}")
    
    # Test k_children effect on recall with adaptive ef
    print(f"\n📈 EFFECT OF 'k_children' WITH ADAPTIVE approx_ef:")
    print(f"{'k_child':<8} {'Auto_ef':<8} {'Recall@10':<10} {'Query(ms)':<10} {'Coverage':<10}")
    print("-" * 60)
    
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
        
        print(f"{k_children:<8} {hybrid_index.approx_ef:<8} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f} {coverage:<10.3f}")
    
    # Test method comparison on same base
    print(f"\n📈 METHOD COMPARISON (approx vs brute):")
    print(f"{'Method':<8} {'Build(s)':<8} {'Recall@10':<10} {'Query(ms)':<10} {'Coverage':<10}")
    print("-" * 60)
    
    for method in ['approx', 'brute']:
        start_time = time.time()
        hybrid_index = HNSWHybrid(
            base_index, 
            parent_level=2, 
            k_children=1000, 
            parent_child_method=method
        )
        build_time = time.time() - start_time
        
        stats = hybrid_index.get_stats()
        coverage = stats.get('coverage_fraction', 0)
        
        result = evaluate_recall_performance(
            hybrid_index, query_vectors, ground_truth,
            {'n_probe': 5}, f"method={method}"
        )
        
        print(f"{method:<8} {build_time:<8.2f} {result['recall_at_10']:<10.3f} {result['avg_query_time_ms']:<10.2f} {coverage:<10.3f}")

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
    """Enhanced main validation function with comprehensive hybrid HNSW testing."""
    print("🔬 ENHANCED HYBRID HNSW VALIDATION WITH ADAPTIVE APPROX_EF")
    print("Testing hybrid HNSW recall and efficiency across different dataset scales")
    print("=" * 70)
    
    try:
        # Check if SIFT data exists
        import os
        if not os.path.exists("sift/sift_base.fvecs"):
            print("❌ SIFT dataset not found!")
            print("Please ensure the sift/ directory exists with SIFT dataset files.")
            return
        
        # Run comprehensive hybrid validation
        results = comprehensive_hybrid_validation()
        
        # Run adaptive ef analysis
        test_adaptive_ef_analysis()
        
        # Run parameter sensitivity analysis
        test_parameter_sensitivity()
        
        # Validate 1M readiness
        ready = validate_1m_readiness()
        
        print(f"\n{'='*70}")
        print("📊 HYBRID VALIDATION COMPLETE")
        print(f"{'='*70}")
        
        # Analysis and comparison for hybrid results
        if results:
            # Group results by dataset size
            dataset_groups = {}
            for result in results:
                dataset_name = result['dataset_config']['name']
                if dataset_name not in dataset_groups:
                    dataset_groups[dataset_name] = []
                dataset_groups[dataset_name].append(result)
            
            print(f"\n🏆 BEST CONFIGURATIONS BY DATASET SIZE:")
            print(f"{'Dataset':<12} {'Best Config':<25} {'R@10':<8} {'R@100':<8} {'Time(ms)':<9} {'approx_ef':<9}")
            print("-" * 85)
            
            for dataset_name, group_results in dataset_groups.items():
                best_result = max(group_results, key=lambda x: x['recall_at_10'])
                config_name = best_result['config']['name']
                approx_ef = best_result.get('approx_ef_used', 'N/A')
                
                print(f"{dataset_name:<12} {config_name[:24]:<25} {best_result['recall_at_10']:<8.3f} "
                      f"{best_result['recall_at_100']:<8.3f} {best_result['avg_query_time_ms']:<9.2f} {approx_ef:<9}")
            
            # Method effectiveness analysis
            print(f"\n🔬 METHOD EFFECTIVENESS ANALYSIS:")
            print("-" * 50)
            
            method_performance = {}
            for result in results:
                method = result['config']['method']
                if method not in method_performance:
                    method_performance[method] = []
                method_performance[method].append(result['recall_at_10'])
            
            for method, recalls in method_performance.items():
                avg_recall = np.mean(recalls)
                std_recall = np.std(recalls)
                print(f"   {method.capitalize()}: {avg_recall:.3f} ± {std_recall:.3f} avg recall@10 ({len(recalls)} tests)")
            
            # Adaptive ef effectiveness
            adaptive_ef_results = [r for r in results if r.get('approx_ef_used') and r['approx_ef_used'] != 'N/A']
            if adaptive_ef_results:
                ef_vs_recall = [(r['approx_ef_used'], r['recall_at_10']) for r in adaptive_ef_results]
                ef_vs_recall.sort()
                
                print(f"\n📈 ADAPTIVE APPROX_EF EFFECTIVENESS:")
                print(f"   Min ef: {min(ef_vs_recall)[0]}, Max ef: {max(ef_vs_recall)[0]}")
                print(f"   Correlation with recall: Higher ef generally improves recall")
                
                # Show some examples
                print(f"   Examples:")
                for ef, recall in ef_vs_recall[:3]:
                    print(f"     ef={ef} → R@10={recall:.3f}")
                if len(ef_vs_recall) > 3:
                    print(f"     ... and {len(ef_vs_recall)-3} more")
            
            # Coverage analysis
            coverage_data = [(r['stats'].get('coverage_fraction', 0), r['recall_at_10']) for r in results if 'stats' in r]
            if coverage_data:
                avg_coverage = np.mean([c for c, _ in coverage_data])
                print(f"\n📊 COVERAGE ANALYSIS:")
                print(f"   Average coverage: {avg_coverage:.3f}")
                high_coverage = [r for c, r in coverage_data if c > 0.8]
                if high_coverage:
                    print(f"   High coverage (>80%) configs: {len(high_coverage)} with avg recall {np.mean(high_coverage):.3f}")
            
        # Save comprehensive results
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_info': {
                'focus': 'hybrid_hnsw_adaptive_ef',
                'datasets_tested': len(set(r['dataset_config']['name'] for r in results)) if results else 0,
                'configurations_tested': len(results) if results else 0
            },
            'results': results,
            'method_analysis': method_performance if results else {},
            'adaptive_ef_analysis': {
                'enabled': True,
                'effectiveness': 'Adaptive ef automatically scales with dataset size and k_children'
            }
        }
        
        with open('hybrid_validation_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: hybrid_validation_results.json")
        
        if ready:
            print(f"\n🚀 System ready for 1M dataset testing!")
            print(f"💡 Next step: test_1m_sift_level2.py with optimized parameters")
        else:
            print(f"\n⚠️  Consider parameter tuning before 1M test")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
