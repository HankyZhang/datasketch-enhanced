"""
HNSW Hybrid Two-Stage Retrieval System

This module implements a hybrid HNSW index structure that transforms a standard HNSW
into a two-stage retrieval system for improved recall evaluation in plaintext environments.

The system consists of:
1. Parent Layer (Coarse Filtering): Nodes from higher HNSW levels as cluster centers
2. Child Layer (Fine Filtering): Precomputed neighbor sets for each parent node

"""

import numpy as np
import time
from typing import Dict, List, Tuple, Set, Optional, Hashable
from collections import defaultdict
import heapq
from math import ceil
import sys
import os

# Add parent directory to path to import hnsw module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW


class HNSWHybrid:
    """
    Hybrid HNSW index that implements a two-stage retrieval system.
    
    This class extracts parent nodes from higher levels of a base HNSW index
    and precomputes child node mappings for efficient two-stage search.
    """
    
    def __init__(
        self,
        base_index: HNSW,
        parent_level: int = 2,
        k_children: int = 1000,
        distance_func: Optional[callable] = None,
        parent_child_method: str = 'approx',
        approx_ef: Optional[int] = None,  # Now auto-computed if None
        diversify_max_assignments: Optional[int] = None,
        repair_min_assignments: Optional[int] = None,
        include_parents_in_results: bool = False,
        overlap_sample: int = 50
    ):
        """
        Initialize the hybrid HNSW index.
        
        Args:
            base_index: The base HNSW index to extract structure from
            parent_level: The HNSW level to extract parent nodes from (default: 2)
            k_children: Number of child nodes to precompute for each parent
            distance_func: Distance function (uses base index's if None)
            approx_ef: Search width for approximate neighbor finding. If None, 
                      auto-computed based on dataset size and k_children
        """
        self.base_index = base_index
        self.parent_level = parent_level
        self.k_children = k_children
        self.distance_func = distance_func or base_index._distance_func

        # Auto-compute approx_ef if not provided
        if approx_ef is None:
            dataset_size = len(base_index)
            # Adaptive formula: ensure ef is large enough for k_children but scales with dataset
            # Base formula: max(k_children * 1.2, min(dataset_size * 0.1, k_children * 2))
            min_ef = max(k_children + 50, int(k_children * 1.2))  # At least 20% more than k_children
            adaptive_ef = min(int(dataset_size * 0.1), int(k_children * 2))  # Scale with dataset but cap at 2*k
            self.approx_ef = max(min_ef, adaptive_ef)
            print(f"Auto-computed approx_ef={self.approx_ef} (dataset_size={dataset_size}, k_children={k_children})")
        else:
            self.approx_ef = approx_ef
            print(f"Using provided approx_ef={self.approx_ef}")

        # Extended configuration
        self.parent_child_method = parent_child_method  # 'approx' | 'brute'
        self.diversify_max_assignments = diversify_max_assignments
        self.repair_min_assignments = repair_min_assignments
        self.include_parents_in_results = include_parents_in_results
        self.overlap_sample = overlap_sample

        # Parent identifiers and vector storage
        self.parent_ids = []
        self.parent_child_map = {}
        self.parent_vectors = {}
        self.child_vectors = {}

        # Vectorized parent matrix (P, D) built after extraction
        self._parent_matrix = None
        self._parent_id_array = None

        # Statistics & diagnostics
        self.stats = {
            'num_parents': 0,
            'num_children': 0,
            'avg_children_per_parent': 0.0,
            'construction_time': 0.0,
            'parent_extraction_time': 0.0,
            'mapping_build_time': 0.0,
            'coverage_fraction': 0.0,
            'avg_search_time_ms': 0.0,
            'avg_candidate_size': 0.0,
            'mapping_method': parent_child_method
        }
        self.search_times = []
        self.candidate_sizes = []
        self._overlap_stats = {}

        # Build the hybrid structure
        self._build_hybrid_structure()
    
    def _build_hybrid_structure(self):
        """Build the parent-child structure from the base HNSW index."""
        print(f"Building hybrid HNSW structure from level {self.parent_level}...")
        start_time = time.time()
        
        # Step 1: Extract parent nodes from the specified level
        t0 = time.time()
        parent_nodes = self._extract_parent_nodes()
        self.stats['parent_extraction_time'] = time.time() - t0
        print(f"Found {len(parent_nodes)} parent nodes at level {self.parent_level}")

        # Step 2: Precompute child mappings for each parent (with method options)
        t1 = time.time()
        self._precompute_child_mappings(parent_nodes)
        self.stats['mapping_build_time'] = time.time() - t1

        # Step 3: Build parent vector index for Stage 1 search (vectorized matrix)
        self._build_parent_index()
        
        # Record statistics
        self.stats['num_parents'] = len(parent_nodes)
        self.stats['num_children'] = len(self.child_vectors)
        self.stats['avg_children_per_parent'] = (
            self.stats['num_children'] / self.stats['num_parents'] 
            if self.stats['num_parents'] > 0 else 0.0
        )
        self.stats['construction_time'] = time.time() - start_time
        
        print(f"Hybrid structure built in {self.stats['construction_time']:.2f}s")
        print(f"Statistics: {self.stats['num_parents']} parents, "
              f"{self.stats['num_children']} children, "
              f"{self.stats['avg_children_per_parent']:.1f} avg children/parent")
    
    def _extract_parent_nodes(self) -> List[Hashable]:
        """Extract parent nodes from the specified HNSW level.

        Falls back to highest existing level if requested level absent.
        """
        if self.parent_level >= len(self.base_index._graphs):
            adjusted = len(self.base_index._graphs) - 1
            print(f"[WARN] Requested parent_level={self.parent_level} unavailable. Using {adjusted} instead.")
            self.parent_level = adjusted

        target_layer = self.base_index._graphs[self.parent_level]
        parent_nodes: List[Hashable] = []
        for node_id in target_layer:
            try:
                if node_id in self.base_index and not self.base_index._nodes[node_id].is_deleted:
                    parent_nodes.append(node_id)
            except Exception:
                # Defensive: Skip problematic nodes
                continue
        self.parent_ids = parent_nodes
        return parent_nodes
    
    def _precompute_child_mappings(self, parent_nodes: List[Hashable]):
        """Precompute child node mappings for each parent with optional diversification & repair."""
        print(f"Precomputing child mappings (method={self.parent_child_method})...")

        assignment_counts: Dict[Hashable, int] = defaultdict(int) if self.diversify_max_assignments else None

        for i, parent_id in enumerate(parent_nodes):
            if i % max(1, len(parent_nodes)//10 or 1) == 0:
                print(f"  Parent {i+1}/{len(parent_nodes)}")

            parent_vector = self.base_index[parent_id]
            self.parent_vectors[parent_id] = parent_vector

            # Generate raw neighbor candidates
            if self.parent_child_method == 'brute':
                # Brute force distances against all nodes in base index
                # Assuming node ids are iterable via base_index._nodes
                raw: List[Tuple[float, Hashable]] = []
                for nid in self.base_index._nodes.keys():
                    if nid == parent_id:
                        continue
                    vec = self.base_index[nid]
                    d = self.distance_func(parent_vector, vec)
                    raw.append((d, nid))
                raw.sort()
                neighbor_ids = [nid for _, nid in raw[: self.k_children + 1]]
            else:  # approx
                # Ensure ef is at least as large as k to get requested number of neighbors
                effective_ef = max(self.approx_ef, self.k_children + 1)
                try:
                    neighbors = self.base_index.query(parent_vector, k=self.k_children + 1, ef=effective_ef)
                except TypeError:
                    neighbors = self.base_index.query(parent_vector, k=self.k_children + 1)
                neighbor_ids = [nid for nid, _ in neighbors]

            # Exclude self
            if parent_id in neighbor_ids:
                neighbor_ids = [nid for nid in neighbor_ids if nid != parent_id]

            # Diversification (limit global assignments per point)
            if assignment_counts is not None:
                accepted: List[Hashable] = []
                skipped: List[Hashable] = []
                limit = self.diversify_max_assignments
                for nid in neighbor_ids:
                    if len(accepted) >= self.k_children:
                        break
                    if assignment_counts[nid] < limit:
                        accepted.append(nid)
                        assignment_counts[nid] += 1
                    else:
                        skipped.append(nid)
                # Backfill if under length
                if len(accepted) < self.k_children:
                    for nid in skipped:
                        if len(accepted) >= self.k_children:
                            break
                        accepted.append(nid)
                child_ids = accepted
            else:
                child_ids = neighbor_ids[: self.k_children]

            for cid in child_ids:
                if cid not in self.child_vectors:
                    try:
                        self.child_vectors[cid] = self.base_index[cid]
                    except Exception:
                        continue

            self.parent_child_map[parent_id] = child_ids

        # Repair phase: ensure every child that appears too few times is assigned to closest parents
        if self.repair_min_assignments and assignment_counts is not None:
            # Find nodes with low coverage (already assigned but less than min)
            low_coverage = [nid for nid, c in assignment_counts.items() if c < self.repair_min_assignments]
            
            # Find completely unassigned nodes (not in assignment_counts at all)
            all_base_nodes = set(self.base_index._nodes.keys()) if hasattr(self.base_index, '_nodes') else set()
            assigned_nodes = set(assignment_counts.keys())
            unassigned_nodes = list(all_base_nodes - assigned_nodes - set(self.parent_ids))  # Exclude parent nodes
            
            # Combine low coverage and unassigned nodes
            nodes_to_repair = low_coverage + unassigned_nodes
            
            if nodes_to_repair:
                print(f"Repairing {len(low_coverage)} low-coverage + {len(unassigned_nodes)} unassigned points (min={self.repair_min_assignments})...")
                # Build parent matrix if absent
                if self._parent_matrix is None:
                    self._build_parent_index()
                P = self._parent_matrix.shape[0]
                for nid in nodes_to_repair:
                    try:
                        vec = self.base_index[nid]
                        diffs = self._parent_matrix - vec
                        dists = np.linalg.norm(diffs, axis=1)
                        order = np.argsort(dists)
                        for idx in order:
                            pid = self.parent_ids[idx]
                            if nid not in self.parent_child_map[pid]:
                                self.parent_child_map[pid].append(nid)
                                assignment_counts[nid] += 1
                                # Add to child_vectors if not already present
                                if nid not in self.child_vectors:
                                    self.child_vectors[nid] = vec
                            if assignment_counts[nid] >= self.repair_min_assignments:
                                break
                    except Exception as e:
                        print(f"[WARN] Failed to repair node {nid}: {e}")
                        continue

        # Coverage & overlap stats
        self._compute_mapping_diagnostics()
    
    def _build_parent_index(self):
        """Vectorize parent vectors for fast Stage 1 distance computation."""
        if not self.parent_vectors:
            return
        # Ensure deterministic ordering consistent with parent_ids
        if not self.parent_ids:
            self.parent_ids = list(self.parent_vectors.keys())
        matrix = []
        for pid in self.parent_ids:
            matrix.append(self.parent_vectors[pid])
        try:
            self._parent_matrix = np.vstack(matrix)
            self._parent_id_array = np.array(self.parent_ids)
        except Exception as e:
            print(f"[WARN] Failed to build parent matrix: {e}")
            self._parent_matrix = None
            self._parent_id_array = None
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        n_probe: int = 10
    ) -> List[Tuple[Hashable, float]]:
        """
        Perform two-stage search on the hybrid index.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            n_probe: Number of parent nodes to probe in Stage 1
            
        Returns:
            List of (node_id, distance) tuples, sorted by distance
        """
        if not self.parent_vectors:
            return []
        start = time.time()
        # Stage 1: Coarse search - find closest parent nodes
        parent_candidates = self._stage1_coarse_search(query_vector, n_probe)

        # Stage 2: Fine search - search within child nodes of selected parents
        results = self._stage2_fine_search(query_vector, parent_candidates, k)

        elapsed = (time.time() - start) * 1000.0
        self.search_times.append(elapsed)
        if results:
            self.stats['avg_search_time_ms'] = float(np.mean(self.search_times))
        return results
    
    def _stage1_coarse_search(
        self, 
        query_vector: np.ndarray, 
        n_probe: int
    ) -> List[Tuple[Hashable, float]]:
        """Stage 1: Find the closest parent nodes using brute force search."""
        parent_distances = []
        
        if self._parent_matrix is not None:
            # Vectorized distance computation
            diffs = self._parent_matrix - query_vector
            dists = np.linalg.norm(diffs, axis=1)
            if n_probe < len(dists):
                idx = np.argpartition(dists, n_probe)[:n_probe]
            else:
                idx = np.arange(len(dists))
            subset = sorted(((dists[i], self.parent_ids[i]) for i in idx), key=lambda x: x[0])
            return subset
        # Fallback loop
        for pid, pvec in self.parent_vectors.items():
            parent_distances.append((self.distance_func(query_vector, pvec), pid))
        parent_distances.sort(key=lambda x: x[0])
        return parent_distances[:n_probe]
    
    def _stage2_fine_search(
        self, 
        query_vector: np.ndarray, 
        parent_candidates: List[Tuple[Hashable, float]], 
        k: int
    ) -> List[Tuple[Hashable, float]]:
        """Stage 2: Search within child nodes of selected parents."""
        candidate_children = set()
        for distance, parent_id in parent_candidates:
            if parent_id in self.parent_child_map:
                candidate_children.update(self.parent_child_map[parent_id])
        if self.include_parents_in_results:
            candidate_children.update(pid for _, pid in parent_candidates)

        if not candidate_children:
            return []

        # Vectorize candidate distances if feasible
        candidate_ids = list(candidate_children)
        vectors = []
        valid_ids = []
        for cid in candidate_ids:
            try:
                vec = self.child_vectors.get(cid)
                if vec is None:
                    vec = self.base_index[cid]
                vectors.append(vec)
                valid_ids.append(cid)
            except Exception:
                continue
        if not vectors:
            return []
        mat = np.vstack(vectors)
        diffs = mat - query_vector
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)[:k]
        # Track candidate size
        self.candidate_sizes.append(len(valid_ids))
        self.stats['avg_candidate_size'] = float(np.mean(self.candidate_sizes))
        return [(valid_ids[i], dists[i]) for i in order]
    
    def update_approx_ef(self, new_ef: int, rebuild_mappings: bool = False):
        """
        Update the approx_ef parameter and optionally rebuild child mappings.
        
        Args:
            new_ef: New value for approx_ef
            rebuild_mappings: If True, rebuild parent-child mappings with new ef
        """
        old_ef = self.approx_ef
        self.approx_ef = new_ef
        print(f"Updated approx_ef from {old_ef} to {new_ef}")
        
        if rebuild_mappings:
            print("Rebuilding parent-child mappings with new approx_ef...")
            start_time = time.time()
            # Clear existing mappings
            self.parent_child_map.clear()
            self.child_vectors.clear()
            
            # Rebuild mappings
            self._precompute_child_mappings(self.parent_ids)
            rebuild_time = time.time() - start_time
            print(f"Mappings rebuilt in {rebuild_time:.2f}s")
            
            # Update statistics
            self.stats['num_children'] = len(self.child_vectors)
            self.stats['avg_children_per_parent'] = (
                self.stats['num_children'] / self.stats['num_parents'] 
                if self.stats['num_parents'] > 0 else 0.0
            )

    def get_recommended_ef(self, target_recall: float = 0.95) -> int:
        """
        Get recommended approx_ef value based on dataset characteristics.
        
        Args:
            target_recall: Target recall level (0.0 to 1.0)
            
        Returns:
            Recommended approx_ef value
        """
        dataset_size = len(self.base_index)
        
        # Heuristic formulas based on target recall
        if target_recall >= 0.95:
            # High recall: need larger search width
            base_multiplier = 2.0
        elif target_recall >= 0.90:
            # Medium-high recall
            base_multiplier = 1.5
        else:
            # Lower recall: can use smaller search width
            base_multiplier = 1.2
            
        # Scale with dataset size and k_children
        recommended = max(
            int(self.k_children * base_multiplier),
            min(int(dataset_size * 0.05), int(self.k_children * 3))
        )
        
        return recommended

    def get_stats(self) -> Dict:
        """Get statistics including overlap diagnostics if available."""
        out = self.stats.copy()
        out.update(self._overlap_stats)
        return out
    
    def get_parent_child_info(self) -> Dict:
        """Get detailed parent-child mapping information."""
        return {
            'parent_child_map': self.parent_child_map,
            'num_parents': len(self.parent_vectors),
            'num_children': len(self.child_vectors),
            'parent_level': self.parent_level,
            'k_children': self.k_children
        }

    # --------------------- Diagnostics ---------------------
    def _compute_mapping_diagnostics(self):
        """Compute coverage and simple overlap statistics for parent-child map."""
        if not self.parent_child_map:
            return
        all_children_sets = [set(v) for v in self.parent_child_map.values() if v]
        if not all_children_sets:
            return
        union_all = set().union(*all_children_sets)
        total_points = len(self.base_index._nodes) if hasattr(self.base_index, '_nodes') else None
        coverage_fraction = len(union_all) / total_points if total_points else 0.0
        self.stats['coverage_fraction'] = coverage_fraction

        # Sample pairwise Jaccard overlaps
        overlaps: List[float] = []
        if len(all_children_sets) > 1:
            import random
            sample = self.overlap_sample
            for _ in range(min(sample, len(all_children_sets) * (len(all_children_sets)-1)//2)):
                a, b = random.sample(all_children_sets, 2)
                inter = len(a & b)
                union = len(a | b)
                if union > 0:
                    overlaps.append(inter / union)
        if overlaps:
            self._overlap_stats = {
                'mean_jaccard_overlap': float(np.mean(overlaps)),
                'median_jaccard_overlap': float(np.median(overlaps))
            }
        else:
            self._overlap_stats = {}


class HNSWEvaluator:
    """
    Evaluator for HNSW hybrid system recall performance.
    """
    
    def __init__(self, dataset: np.ndarray, query_set: np.ndarray, query_ids: List[Hashable]):
        """
        Initialize the evaluator.
        
        Args:
            dataset: Full dataset vectors (shape: [n_vectors, dim])
            query_set: Query vectors (shape: [n_queries, dim])
            query_ids: IDs for query vectors
        """
        self.dataset = dataset
        self.query_set = query_set
        self.query_ids = query_ids
        
        # Ground truth will be computed lazily
        self._ground_truth: Optional[Dict[Hashable, List[Tuple[Hashable, float]]]] = None
    
    def compute_ground_truth(self, k: int, distance_func: callable) -> Dict[Hashable, List[Tuple[Hashable, float]]]:
        """
        Compute ground truth using brute force search.
        
        Args:
            k: Number of nearest neighbors to find
            distance_func: Distance function to use
            
        Returns:
            Dictionary mapping query_id to list of (neighbor_id, distance) tuples
        """
        print(f"Computing ground truth for {len(self.query_set)} queries...")
        start_time = time.time()
        
        ground_truth = {}
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            if i % 1000 == 0:
                print(f"Processing query {i+1}/{len(self.query_set)}")
            
            # Calculate distances to all dataset vectors
            distances = []
            for j, dataset_vector in enumerate(self.dataset):
                distance = distance_func(query_vector, dataset_vector)
                distances.append((distance, j))
            
            # Sort by distance and take top k
            distances.sort()
            # Convert from (distance, neighbor_id) to (neighbor_id, distance) format
            ground_truth[query_id] = [(neighbor_id, distance) for distance, neighbor_id in distances[:k]]
        
        self._ground_truth = ground_truth
        print(f"Ground truth computed in {time.time() - start_time:.2f}s")
        return ground_truth
    
    def evaluate_recall(
        self, 
        hybrid_index: HNSWHybrid, 
        k: int, 
        n_probe: int,
        ground_truth: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate recall performance of the hybrid index.
        
        Args:
            hybrid_index: The hybrid HNSW index to evaluate
            k: Number of results to return
            n_probe: Number of parent nodes to probe
            ground_truth: Precomputed ground truth (optional)
            
        Returns:
            Dictionary containing recall metrics and timing information
        """
        if ground_truth is None:
            if self._ground_truth is None:
                ground_truth = self.compute_ground_truth(k, hybrid_index.distance_func)
            else:
                ground_truth = self._ground_truth
        
        print(f"Evaluating recall for {len(self.query_set)} queries...")
        start_time = time.time()
        
        total_correct = 0
        total_expected = len(self.query_set) * k
        query_times = []
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            if i % 1000 == 0:
                print(f"Evaluating query {i+1}/{len(self.query_set)}")
            
            # Time the query
            query_start = time.time()
            results = hybrid_index.search(query_vector, k=k, n_probe=n_probe)
            query_time = time.time() - query_start
            query_times.append(query_time)
            
            # Extract result IDs
            result_ids = set(neighbor_id for neighbor_id, _ in results)
            
            # Get ground truth IDs
            gt_ids = set(neighbor_id for neighbor_id, _ in ground_truth[query_id])
            
            # Count correct matches
            correct = len(result_ids.intersection(gt_ids))
            total_correct += correct
        
        # Calculate metrics
        recall_at_k = total_correct / total_expected
        avg_query_time = np.mean(query_times)
        total_evaluation_time = time.time() - start_time
        
        return {
            'recall_at_k': recall_at_k,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'avg_query_time_ms': avg_query_time * 1000,
            'total_evaluation_time': total_evaluation_time,
            'query_times': query_times,
            'k': k,
            'n_probe': n_probe,
            'hybrid_stats': hybrid_index.get_stats()
        }
    
    def parameter_sweep(
        self, 
        hybrid_index: HNSWHybrid, 
        k_values: List[int], 
        n_probe_values: List[int],
        ground_truth: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform parameter sweep to find optimal settings.
        
        Args:
            hybrid_index: The hybrid HNSW index to evaluate
            k_values: List of k values to test
            n_probe_values: List of n_probe values to test
            ground_truth: Precomputed ground truth (optional)
            
        Returns:
            List of evaluation results for each parameter combination
        """
        results = []
        
        for k in k_values:
            for n_probe in n_probe_values:
                print(f"\nEvaluating k={k}, n_probe={n_probe}")
                result = self.evaluate_recall(hybrid_index, k, n_probe, ground_truth)
                results.append(result)
        
        return results


def create_synthetic_dataset(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """Create a synthetic dataset for testing."""
    np.random.seed(seed)
    return np.random.randn(n_vectors, dim).astype(np.float32)


def create_query_set(dataset: np.ndarray, n_queries: int, seed: int = 123) -> Tuple[np.ndarray, List[int]]:
    """Create a query set by sampling from the dataset."""
    np.random.seed(seed)
    n_dataset = len(dataset)
    query_indices = np.random.choice(n_dataset, size=n_queries, replace=False)
    query_vectors = dataset[query_indices]
    return query_vectors, query_indices.tolist()


if __name__ == "__main__":
    # Example usage
    print("HNSW Hybrid System - Example Usage")
    
    # Create synthetic data
    print("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(10000, 128)  # 10K vectors for demo
    query_vectors, query_ids = create_query_set(dataset, 100)  # 100 queries for demo
    
    # Build base HNSW index
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    # Insert all vectors except queries
    for i, vector in enumerate(dataset):
        if i not in query_ids:  # Exclude query vectors from index
            base_index.insert(i, vector)
    
    # Build hybrid index
    print("Building hybrid HNSW index...")
    hybrid_index = HNSWHybrid(
        base_index=base_index,
        parent_level=2,
        k_children=500
    )
    
    # Evaluate
    print("Evaluating recall...")
    evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
    
    # Compute ground truth
    ground_truth = evaluator.compute_ground_truth(k=10, distance_func=distance_func)
    
    # Evaluate recall
    result = evaluator.evaluate_recall(hybrid_index, k=10, n_probe=5, ground_truth=ground_truth)
    
    print(f"\nResults:")
    print(f"Recall@10: {result['recall_at_k']:.4f}")
    print(f"Average query time: {result['avg_query_time_ms']:.2f} ms")
    print(f"Hybrid stats: {result['hybrid_stats']}")
