#!/usr/bin/env python3
"""
Debug HNSW Recall Issues - Why is recall always 1.0?

This script investigates potential issues in recall calculation:
1. Ground truth computation problems
2. Index ID misalignment
3. Dataset size effects
4. HNSW parameter sensitivity
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

def debug_ground_truth_computation():
    """Debug ground truth computation to identify issues."""
    print("🔍 DEBUGGING GROUND TRUTH COMPUTATION")
    print("=" * 50)
    
    # Load small subset for debugging
    base_vectors = read_fvecs("sift/sift_base.fvecs", 1000)  # Only 1K vectors
    query_vectors = read_fvecs("sift/sift_query.fvecs", 5)   # Only 5 queries
    
    print(f"Dataset: {len(base_vectors)} base, {len(query_vectors)} queries")
    print(f"Vector dimension: {base_vectors.shape[1]}")
    
    # Compute ground truth manually
    print("\n🎯 Computing manual ground truth...")
    manual_gt = []
    
    for i, query in enumerate(query_vectors):
        print(f"\nQuery {i}:")
        distances = []
        for j, base_vec in enumerate(base_vectors):
            dist = np.linalg.norm(query - base_vec)
            distances.append((dist, j))
        
        distances.sort()
        gt_ids = [idx for _, idx in distances[:10]]
        manual_gt.append(gt_ids)
        
        print(f"  Top 3 nearest: {gt_ids[:3]} with distances: {[round(distances[k][0], 3) for k in range(3)]}")
    
    # Now test HNSW search
    print("\n🔬 Testing HNSW search...")
    from hnsw import HNSW
    
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    # Build HNSW with low quality to see if recall drops
    hnsw_index = HNSW(distance_func=distance_func, m=4, ef_construction=50)  # Low quality
    hnsw_index.update(dataset)
    
    print("\n📊 HNSW Results vs Manual Ground Truth:")
    for i, query in enumerate(query_vectors):
        # Test with low ef to potentially reduce recall
        hnsw_results = hnsw_index.query(query, k=10, ef=10)  # Very low ef
        hnsw_ids = [rid for rid, _ in hnsw_results]
        
        gt_ids = manual_gt[i]
        
        print(f"\nQuery {i}:")
        print(f"  Manual GT top 5: {gt_ids[:5]}")
        print(f"  HNSW top 5:      {hnsw_ids[:5]}")
        
        # Calculate recall
        intersection = len(set(hnsw_ids[:10]) & set(gt_ids[:10]))
        recall = intersection / 10
        print(f"  Recall@10: {recall:.3f} ({intersection}/10 matches)")
        
        # Check if exact same
        if hnsw_ids[:10] == gt_ids[:10]:
            print("  ⚠️  EXACT MATCH - This might indicate a problem!")
        
        # Show distance comparison
        hnsw_distances = [dist for _, dist in hnsw_results[:5]]
        manual_distances = [np.linalg.norm(query - base_vectors[idx]) for idx in gt_ids[:5]]
        print(f"  HNSW distances:   {[round(d, 3) for d in hnsw_distances]}")
        print(f"  Manual distances: {[round(d, 3) for d in manual_distances]}")

def test_with_real_sift_groundtruth():
    """Test using the actual SIFT ground truth file."""
    print("\n🎯 TESTING WITH REAL SIFT GROUND TRUTH")
    print("=" * 50)
    
    try:
        # Load actual SIFT ground truth
        ground_truth = read_ivecs("sift/sift_groundtruth.ivecs", 10)
        base_vectors = read_fvecs("sift/sift_base.fvecs", 10000)  # 10K base
        query_vectors = read_fvecs("sift/sift_query.fvecs", 10)   # 10 queries
        
        print(f"Real GT shape: {ground_truth.shape}")
        print(f"Base vectors: {len(base_vectors)}")
        print(f"Query vectors: {len(query_vectors)}")
        
        from hnsw import HNSW
        distance_func = lambda x, y: np.linalg.norm(x - y)
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        
        # Build HNSW
        hnsw_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
        hnsw_index.update(dataset)
        
        print("\n📊 Real Ground Truth vs HNSW:")
        total_recall = 0
        
        for i in range(len(query_vectors)):
            query = query_vectors[i]
            
            # HNSW search
            hnsw_results = hnsw_index.query(query, k=100, ef=50)
            hnsw_ids = [rid for rid, _ in hnsw_results]
            
            # Real ground truth (only indices that exist in our subset)
            real_gt = ground_truth[i]
            valid_gt = [idx for idx in real_gt if idx < len(base_vectors)][:100]
            
            print(f"\nQuery {i}:")
            print(f"  Real GT top 5: {valid_gt[:5]}")
            print(f"  HNSW top 5:    {hnsw_ids[:5]}")
            
            # Calculate recall@10
            k = min(10, len(valid_gt), len(hnsw_ids))
            intersection = len(set(hnsw_ids[:k]) & set(valid_gt[:k]))
            recall = intersection / k if k > 0 else 0
            total_recall += recall
            
            print(f"  Recall@{k}: {recall:.3f} ({intersection}/{k} matches)")
            
            if recall == 1.0:
                print("  ⚠️  Perfect recall - checking distances...")
                # Verify distances
                for j in range(min(3, len(hnsw_ids))):
                    hnsw_dist = np.linalg.norm(query - base_vectors[hnsw_ids[j]])
                    if j < len(valid_gt):
                        gt_dist = np.linalg.norm(query - base_vectors[valid_gt[j]])
                        print(f"    Pos {j}: HNSW={hnsw_dist:.4f}, GT={gt_dist:.4f}")
        
        avg_recall = total_recall / len(query_vectors)
        print(f"\n📈 Average Recall@10: {avg_recall:.3f}")
        
        if avg_recall > 0.95:
            print("🚨 SUSPICIOUSLY HIGH RECALL!")
            print("Possible causes:")
            print("1. Dataset too small - HNSW works perfectly on small data")
            print("2. Index IDs are aligned with ground truth")
            print("3. ef parameter too high for the dataset size")
            print("4. Ground truth subset issue")
        
    except FileNotFoundError as e:
        print(f"❌ SIFT ground truth file not found: {e}")
        print("Using subset-based ground truth instead...")

def test_recall_with_noise():
    """Test recall with intentionally degraded HNSW parameters."""
    print("\n🔬 TESTING RECALL WITH DEGRADED PARAMETERS")
    print("=" * 50)
    
    base_vectors = read_fvecs("sift/sift_base.fvecs", 5000)
    query_vectors = read_fvecs("sift/sift_query.fvecs", 10)
    
    print(f"Dataset: {len(base_vectors)} base, {len(query_vectors)} queries")
    
    # Compute ground truth
    print("Computing ground truth...")
    ground_truth = []
    for query in query_vectors:
        distances = np.linalg.norm(base_vectors - query, axis=1)
        nearest = np.argsort(distances)[:100]
        ground_truth.append(nearest)
    
    from hnsw import HNSW
    distance_func = lambda x, y: np.linalg.norm(x - y)
    dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
    
    # Test different parameter combinations
    test_configs = [
        {"m": 2, "ef_construction": 10, "ef": 5, "name": "Very Low Quality"},
        {"m": 4, "ef_construction": 20, "ef": 10, "name": "Low Quality"},  
        {"m": 8, "ef_construction": 50, "ef": 20, "name": "Medium Quality"},
        {"m": 16, "ef_construction": 200, "ef": 50, "name": "High Quality"},
        {"m": 16, "ef_construction": 200, "ef": 200, "name": "Very High Quality"},
    ]
    
    print(f"\n📊 Parameter vs Recall Analysis:")
    print(f"{'Config':<20} {'Recall@1':<10} {'Recall@10':<11} {'Query(ms)':<10}")
    print("-" * 55)
    
    for config in test_configs:
        # Build index
        hnsw_index = HNSW(
            distance_func=distance_func, 
            m=config['m'], 
            ef_construction=config['ef_construction']
        )
        hnsw_index.update(dataset)
        
        # Test recall
        recalls_1 = []
        recalls_10 = []
        query_times = []
        
        for i, query in enumerate(query_vectors):
            start_time = time.time()
            results = hnsw_index.query(query, k=100, ef=config['ef'])
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)
            
            result_ids = [rid for rid, _ in results]
            gt_ids = ground_truth[i]
            
            # Calculate recall
            recall_1 = 1.0 if result_ids[0] == gt_ids[0] else 0.0
            intersection_10 = len(set(result_ids[:10]) & set(gt_ids[:10]))
            recall_10 = intersection_10 / 10
            
            recalls_1.append(recall_1)
            recalls_10.append(recall_10)
        
        avg_recall_1 = np.mean(recalls_1)
        avg_recall_10 = np.mean(recalls_10)
        avg_query_time = np.mean(query_times)
        
        print(f"{config['name']:<20} {avg_recall_1:<10.3f} {avg_recall_10:<11.3f} {avg_query_time:<10.2f}")
        
        # If still perfect recall with terrible parameters, something's wrong
        if avg_recall_10 > 0.98 and config['name'] == "Very Low Quality":
            print(f"🚨 Still perfect recall with terrible parameters!")
            print(f"   This suggests the dataset is too small or there's an indexing issue")

def identify_recall_issues():
    """Comprehensive analysis to identify why recall is always 1.0."""
    print("🔍 COMPREHENSIVE RECALL ISSUE ANALYSIS")
    print("=" * 60)
    
    # Check 1: Dataset size effect
    print("\n1️⃣ DATASET SIZE EFFECT TEST")
    print("-" * 30)
    
    for size in [100, 500, 1000, 5000]:
        print(f"\nTesting with {size} vectors...")
        base_vectors = read_fvecs("sift/sift_base.fvecs", size)
        query_vectors = read_fvecs("sift/sift_query.fvecs", 3)
        
        from hnsw import HNSW
        distance_func = lambda x, y: np.linalg.norm(x - y)
        dataset = {i: base_vectors[i] for i in range(len(base_vectors))}
        
        # Use poor parameters
        hnsw_index = HNSW(distance_func=distance_func, m=4, ef_construction=20)
        hnsw_index.update(dataset)
        
        # Test first query
        query = query_vectors[0]
        results = hnsw_index.query(query, k=10, ef=5)  # Very low ef
        
        # Compute true nearest neighbors
        distances = np.linalg.norm(base_vectors - query, axis=1)
        true_nearest = np.argsort(distances)[:10]
        hnsw_nearest = [rid for rid, _ in results]
        
        intersection = len(set(hnsw_nearest) & set(true_nearest))
        recall = intersection / 10
        
        print(f"  Size {size}: Recall = {recall:.3f}")
        
        if recall < 1.0:
            print(f"  ✅ Found imperfect recall at size {size}!")
            break
    else:
        print("  🚨 All sizes show perfect recall - issue confirmed!")

def main():
    """Main debugging function."""
    print("🐛 HNSW RECALL = 1.0 DEBUGGING")
    print("Investigating why HNSW always shows perfect recall")
    print("=" * 60)
    
    try:
        import os
        if not os.path.exists("sift/sift_base.fvecs"):
            print("❌ SIFT dataset not found!")
            return
        
        # Run debugging tests
        debug_ground_truth_computation()
        test_with_real_sift_groundtruth()
        test_recall_with_noise()
        identify_recall_issues()
        
        print(f"\n{'='*60}")
        print("🔍 DEBUGGING SUMMARY")
        print(f"{'='*60}")
        print("Possible causes of recall = 1.0:")
        print("1. 🔢 Dataset too small - HNSW works perfectly on small datasets")
        print("2. 🎯 Ground truth computation aligned with search space")
        print("3. ⚙️  Parameters too high quality for dataset size")
        print("4. 🔗 Index ID alignment creating artificial perfect matches")
        print("5. 📊 Test queries are too similar to base vectors")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("• Use larger datasets (50K+ vectors)")
        print("• Test with lower ef parameters (ef=10-20)")
        print("• Use real SIFT ground truth instead of computed")
        print("• Add random noise to test robustness")
        print("• Test with queries from different distribution")
        
    except Exception as e:
        print(f"❌ Debugging failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
