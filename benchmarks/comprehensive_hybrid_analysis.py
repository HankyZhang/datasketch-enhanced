#!/usr/bin/env python3
"""
Comprehensive test showing why Hybrid HNSW has low recall and how to fix it
"""

import sys
import os
sys.path.append('.')

import numpy as np
from datasketch import HNSW

class ComprehensiveHybridTest:
    """Test class that demonstrates the fundamental Hybrid HNSW issues"""
    
    def __init__(self, k_children=200, n_probe=5):
        self.k_children = k_children
        self.n_probe = n_probe
        
    def create_clustered_dataset(self, n_clusters=5, vectors_per_cluster=200, dim=32):
        """Create a proper clustered dataset for testing"""
        np.random.seed(42)  # Reproducible results
        dataset = []
        
        print(f"Creating {n_clusters} clusters with {vectors_per_cluster} vectors each...")
        
        for cluster_id in range(n_clusters):
            # Create well-separated cluster centers
            center = np.random.random(dim) * 10 + cluster_id * 5
            
            for i in range(vectors_per_cluster):
                noise = np.random.normal(0, 0.1, dim)  # Small noise within cluster
                vector = center + noise
                dataset.append(vector)
        
        return np.array(dataset)
    
    def build_hnsw_index(self, dataset):
        """Build standard HNSW index"""
        index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
        
        for i, vector in enumerate(dataset):
            index.insert(i, vector)
            
        return index
    
    def get_ground_truth(self, dataset, query_vector, k=10):
        """Get ground truth nearest neighbors"""
        distances = []
        for i, vector in enumerate(dataset):
            dist = np.linalg.norm(query_vector - vector)
            distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        return [idx for idx, _ in distances[:k]]
    
    def test_original_hybrid(self, dataset, query_vector, query_id):
        """Test original Hybrid HNSW approach"""
        print("\n" + "="*60)
        print("🔴 ORIGINAL HYBRID APPROACH")
        print("="*60)
        
        # Build base HNSW
        hnsw = self.build_hnsw_index(dataset)
        
        # Extract parent nodes from top level
        top_level = len(hnsw._graphs) - 1
        while top_level > 0 and len(hnsw._graphs[top_level]._graph) < 3:
            top_level -= 1
        
        parent_ids = list(hnsw._graphs[top_level]._graph.keys())
        print(f"Extracted {len(parent_ids)} parent nodes from level {top_level}")
        
        # Build parent-child mapping using HNSW query
        parent_child_map = {}
        total_coverage = set()
        
        for parent_id in parent_ids:
            parent_vector = dataset[parent_id]
            # THE PROBLEM: Using HNSW query for children
            neighbors = hnsw.query(parent_vector, k=self.k_children + 1)
            children = [nid for nid, _ in neighbors if nid != parent_id][:self.k_children]
            parent_child_map[parent_id] = children
            total_coverage.update(children)
            print(f"Parent {parent_id}: {len(children)} children")
        
        coverage = len(total_coverage) / len(dataset) * 100
        query_in_coverage = query_id in total_coverage
        
        print(f"Total coverage: {coverage:.1f}% ({len(total_coverage)}/{len(dataset)})")
        print(f"Query {query_id} in coverage: {query_in_coverage}")
        
        # Perform hybrid search
        # Find closest parents (using simple distance)
        parent_distances = []
        for parent_id in parent_ids:
            dist = np.linalg.norm(query_vector - dataset[parent_id])
            parent_distances.append((parent_id, dist))
        
        parent_distances.sort(key=lambda x: x[1])
        selected_parents = [pid for pid, _ in parent_distances[:self.n_probe]]
        print(f"Selected parents: {selected_parents}")
        
        # Collect candidates
        candidates = set()
        for parent_id in selected_parents:
            candidates.update(parent_child_map[parent_id])
        
        print(f"Candidate set size: {len(candidates)}")
        print(f"Query in candidates: {query_id in candidates}")
        
        # Search among candidates
        if candidates:
            candidate_distances = []
            for cid in candidates:
                dist = np.linalg.norm(query_vector - dataset[cid])
                candidate_distances.append((cid, dist))
            
            candidate_distances.sort(key=lambda x: x[1])
            results = [cid for cid, _ in candidate_distances[:10]]
        else:
            results = []
        
        return results, coverage, query_in_coverage
    
    def test_fixed_hybrid(self, dataset, query_vector, query_id):
        """Test fixed Hybrid HNSW approach"""
        print("\n" + "="*60)
        print("🟢 FIXED HYBRID APPROACH")
        print("="*60)
        
        # Build base HNSW (same as before)
        hnsw = self.build_hnsw_index(dataset)
        
        # Extract parent nodes
        top_level = len(hnsw._graphs) - 1
        while top_level > 0 and len(hnsw._graphs[top_level]._graph) < 3:
            top_level -= 1
        
        parent_ids = list(hnsw._graphs[top_level]._graph.keys())
        print(f"Extracted {len(parent_ids)} parent nodes from level {top_level}")
        
        # Build parent-child mapping using CONSISTENT distance calculation
        parent_child_map = {}
        total_coverage = set()
        
        for parent_id in parent_ids:
            parent_vector = dataset[parent_id]
            
            # THE FIX: Use consistent distance calculation
            distances = []
            for i, vector in enumerate(dataset):
                if i != parent_id:
                    dist = np.linalg.norm(parent_vector - vector)
                    distances.append((i, dist))
            
            distances.sort(key=lambda x: x[1])
            children = [idx for idx, _ in distances[:self.k_children]]
            parent_child_map[parent_id] = children
            total_coverage.update(children)
            print(f"Parent {parent_id}: {len(children)} children")
        
        coverage = len(total_coverage) / len(dataset) * 100
        query_in_coverage = query_id in total_coverage
        
        print(f"Total coverage: {coverage:.1f}% ({len(total_coverage)}/{len(dataset)})")
        print(f"Query {query_id} in coverage: {query_in_coverage}")
        
        # Perform hybrid search (same as before)
        parent_distances = []
        for parent_id in parent_ids:
            dist = np.linalg.norm(query_vector - dataset[parent_id])
            parent_distances.append((parent_id, dist))
        
        parent_distances.sort(key=lambda x: x[1])
        selected_parents = [pid for pid, _ in parent_distances[:self.n_probe]]
        print(f"Selected parents: {selected_parents}")
        
        # Collect candidates
        candidates = set()
        for parent_id in selected_parents:
            candidates.update(parent_child_map[parent_id])
        
        print(f"Candidate set size: {len(candidates)}")
        print(f"Query in candidates: {query_id in candidates}")
        
        # Search among candidates
        if candidates:
            candidate_distances = []
            for cid in candidates:
                dist = np.linalg.norm(query_vector - dataset[cid])
                candidate_distances.append((cid, dist))
            
            candidate_distances.sort(key=lambda x: x[1])
            results = [cid for cid, _ in candidate_distances[:10]]
        else:
            results = []
        
        return results, coverage, query_in_coverage
    
    def run_complete_test(self):
        """Run complete test comparing all approaches"""
        print("🔍 COMPREHENSIVE HYBRID HNSW ANALYSIS")
        print("=" * 65)
        
        # Create test dataset
        dataset = self.create_clustered_dataset(n_clusters=5, vectors_per_cluster=200)
        
        # Use first vector from first cluster as query
        query_id = 0
        query_vector = dataset[query_id]
        
        print(f"\\nQuery: Vector {query_id} (from cluster 0)")
        print("Expected: Should find other vectors from cluster 0 (IDs 0-199)")
        
        # Get ground truth
        ground_truth = self.get_ground_truth(dataset, query_vector, k=10)
        print(f"Ground truth top-10: {ground_truth}")
        
        # Test baseline HNSW
        print("\\n" + "="*60)
        print("🟦 BASELINE HNSW (Reference)")
        print("="*60)
        
        hnsw_baseline = self.build_hnsw_index(dataset)
        baseline_results = hnsw_baseline.query(query_vector, k=10)
        baseline_neighbors = [nid for nid, _ in baseline_results]
        baseline_recall = len(set(baseline_neighbors) & set(ground_truth)) / 10
        
        print(f"Baseline results: {baseline_neighbors}")
        print(f"Baseline recall@10: {baseline_recall:.3f} ({baseline_recall*100:.1f}%)")
        
        # Test original hybrid
        orig_results, orig_coverage, orig_query_covered = self.test_original_hybrid(
            dataset, query_vector, query_id)
        orig_recall = len(set(orig_results) & set(ground_truth)) / 10
        
        print(f"Original results: {orig_results}")
        print(f"Original recall@10: {orig_recall:.3f} ({orig_recall*100:.1f}%)")
        
        # Test fixed hybrid
        fixed_results, fixed_coverage, fixed_query_covered = self.test_fixed_hybrid(
            dataset, query_vector, query_id)
        fixed_recall = len(set(fixed_results) & set(ground_truth)) / 10
        
        print(f"Fixed results: {fixed_results}")
        print(f"Fixed recall@10: {fixed_recall:.3f} ({fixed_recall*100:.1f}%)")
        
        # Summary
        print("\\n" + "="*60)
        print("📊 FINAL COMPARISON")
        print("="*60)
        print(f"{'Method':<20} {'Coverage':<12} {'Query Covered':<15} {'Recall@10':<12}")
        print("-" * 60)
        print(f"{'Baseline HNSW':<20} {'100.0%':<12} {'Yes':<15} {baseline_recall*100:<12.1f}%")
        print(f"{'Original Hybrid':<20} {orig_coverage:<12.1f}% {str(orig_query_covered):<15} {orig_recall*100:<12.1f}%")
        print(f"{'Fixed Hybrid':<20} {fixed_coverage:<12.1f}% {str(fixed_query_covered):<15} {fixed_recall*100:<12.1f}%")
        
        print("\\n🎯 KEY INSIGHTS:")
        print("1. Original Hybrid: Low coverage due to HNSW query inconsistency")
        print("2. Fixed Hybrid: Better coverage with consistent distance calculation")
        print("3. Both hybrid approaches still limited by parent node quality")
        print("4. HNSW layers are not optimal cluster centers")
        
        return {
            'baseline_recall': baseline_recall,
            'original_recall': orig_recall,
            'fixed_recall': fixed_recall,
            'original_coverage': orig_coverage,
            'fixed_coverage': fixed_coverage
        }

def main():
    tester = ComprehensiveHybridTest(k_children=150, n_probe=3)
    results = tester.run_complete_test()
    
    print("\\n🚀 CONCLUSION:")
    improvement = results['fixed_recall'] - results['original_recall']
    print(f"Fixed approach improves recall by {improvement*100:.1f} percentage points")
    print("However, both approaches are still limited by the fundamental")
    print("assumption that HNSW layer nodes make good cluster centers.")

if __name__ == "__main__":
    main()
