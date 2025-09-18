"""
Method 3: K-Means-based Two-Stage HNSW System

This module implements a two-stage retrieval system using K-Means clustering for parent 
discovery and HNSW for child assignment. The system consists of three phases:

Phase 1: Standard HNSW Index (reuse existing base_index)
Phase 2: K-Means Parent Discovery + HNSW Child Assignment
Phase 3: Two-Stage Search (K-Means → HNSW children)

Key Features:
- K-Means centroids as parent nodes instead of HNSW levels
- HNSW-based neighbor assignment for each centroid
- Configurable clustering and search parameters
- Compatible evaluation framework with Methods 1 & 2
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Hashable
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hnsw.hnsw import HNSW
from kmeans.kmeans import KMeans


class KMeansHNSW:
    """
    K-Means-based Two-Stage HNSW System.
    
    This system uses K-Means clustering to identify parent nodes (centroids) and 
    HNSW to find children for each parent, creating an efficient two-stage search.
    """
    
    def __init__(
        self,
        base_index: HNSW,
        n_clusters: int = 100,
        k_children: int = 1000,
        distance_func: Optional[callable] = None,
        child_search_ef: Optional[int] = None,
        kmeans_params: Optional[Dict] = None,
        include_centroids_in_results: bool = False,
        diversify_max_assignments: Optional[int] = None,
        repair_min_assignments: Optional[int] = None,
        overlap_sample: int = 50
    ):
        """
        Initialize the K-Means HNSW system.
        
        Args:
            base_index: The base HNSW index (Phase 1)
            n_clusters: Number of K-Means clusters (parent nodes)
            k_children: Number of child nodes per parent
            distance_func: Distance function (uses base index's if None)
            child_search_ef: Search width for HNSW child finding
            kmeans_params: Additional K-Means parameters
            include_centroids_in_results: Whether to include centroids in search results
            diversify_max_assignments: Maximum assignments per child for diversification
            repair_min_assignments: Minimum assignments per child for repair
            overlap_sample: Sample size for overlap statistics
        """
        self.base_index = base_index
        self.n_clusters = n_clusters
        self.k_children = k_children
        self.distance_func = distance_func or base_index._distance_func
        
        # Auto-compute child_search_ef if not provided
        if child_search_ef is None:
            dataset_size = len(base_index)
            # Adaptive formula similar to hybrid HNSW
            min_ef = max(k_children + 50, int(k_children * 1.2))
            adaptive_ef = min(int(dataset_size * 0.1), int(k_children * 2))
            self.child_search_ef = max(min_ef, adaptive_ef)
        else:
            self.child_search_ef = child_search_ef
        
        # K-Means parameters - optimized for speed
        default_kmeans_params = {
            'max_iters': 100,  # Reduced from 300 for speed
            'tol': 1e-3,       # Relaxed tolerance for faster convergence
            'n_init': 3,       # Reduced from 10 for speed
            'init': 'k-means++',
            'random_state': 42,
            'verbose': False   # Set to False for speed
        }
        if kmeans_params:
            default_kmeans_params.update(kmeans_params)
        self.kmeans_params = default_kmeans_params
        
        # Configuration
        self.include_centroids_in_results = include_centroids_in_results
        self.diversify_max_assignments = diversify_max_assignments
        self.repair_min_assignments = repair_min_assignments
        self.overlap_sample = overlap_sample
        
        # Phase 2 components
        self.kmeans_model = None
        self.centroids = None  # Shape: (n_clusters, dim)
        self.centroid_ids = []  # Virtual IDs for centroids
        self.parent_child_map = {}  # centroid_id -> List[child_id]
        self.child_vectors = {}  # child_id -> vector
        
        # Vectorized matrices for fast search
        self._centroid_matrix = None
        self._centroid_id_array = None
        
        # Statistics
        self.stats = {
            'n_clusters': n_clusters,
            'k_children': k_children,
            'child_search_ef': self.child_search_ef,
            'kmeans_fit_time': 0.0,
            'child_mapping_time': 0.0,
            'total_construction_time': 0.0,
            'num_children': 0,
            'avg_children_per_centroid': 0.0,
            'coverage_fraction': 0.0,
            'avg_search_time_ms': 0.0,
            'avg_candidate_size': 0.0
        }
        self.search_times = []
        self.candidate_sizes = []
        self._overlap_stats = {}
        
        # Build the system
        self._build_kmeans_hnsw_system()
    
    def _build_kmeans_hnsw_system(self):
        """Build the complete K-Means HNSW system (Phase 2)."""
        print(f"Building K-Means HNSW system with {self.n_clusters} clusters...")
        start_time = time.time()
        
        # Step 1: Extract dataset vectors from base HNSW
        dataset_vectors = self._extract_dataset_vectors()
        print(f"Extracted {len(dataset_vectors)} vectors from base HNSW index")
        
        # Step 2: Perform K-Means clustering
        t0 = time.time()
        self._perform_kmeans_clustering(dataset_vectors)
        self.stats['kmeans_fit_time'] = time.time() - t0
        print(f"K-Means clustering completed in {self.stats['kmeans_fit_time']:.2f}s")
        
        # Step 3: Find children for each centroid using HNSW
        t1 = time.time()
        self._assign_children_via_hnsw()
        self.stats['child_mapping_time'] = time.time() - t1
        print(f"Child assignment completed in {self.stats['child_mapping_time']:.2f}s")
        
        # Step 4: Build vectorized centroid matrix for fast search
        self._build_centroid_index()
        
        # Record final statistics
        self.stats['total_construction_time'] = time.time() - start_time
        self.stats['num_children'] = len(self.child_vectors)
        self.stats['avg_children_per_centroid'] = (
            self.stats['num_children'] / self.n_clusters if self.n_clusters > 0 else 0.0
        )
        
        # Compute diagnostics
        self._compute_mapping_diagnostics()
        
        print(f"K-Means HNSW system built in {self.stats['total_construction_time']:.2f}s")
        print(f"Statistics: {self.n_clusters} clusters, "
              f"{self.stats['num_children']} children, "
              f"{self.stats['avg_children_per_centroid']:.1f} avg children/cluster")
    
    def _extract_dataset_vectors(self) -> np.ndarray:
        """Extract all vectors from the base HNSW index."""
        vectors = []
        for key in self.base_index:
            if key in self.base_index:  # Check not soft-deleted
                vectors.append(self.base_index[key])
        
        if not vectors:
            raise ValueError("No vectors found in base HNSW index")
        
        return np.vstack(vectors)
    
    def _perform_kmeans_clustering(self, dataset_vectors: np.ndarray):
        """Perform K-Means clustering to identify parent centroids."""
        print(f"Running K-Means clustering with {self.n_clusters} clusters...")
        
        # Initialize K-Means model
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            **self.kmeans_params
        )
        
        # Fit the model
        self.kmeans_model.fit(dataset_vectors)
        
        # Extract centroids and create virtual IDs
        self.centroids = self.kmeans_model.cluster_centers_
        self.centroid_ids = [f"centroid_{i}" for i in range(self.n_clusters)]
        
        print(f"K-Means completed with inertia: {self.kmeans_model.inertia_:.2f}")
        
        # Print cluster info
        cluster_info = self.kmeans_model.get_cluster_info()
        print(f"Cluster sizes - Avg: {cluster_info['avg_cluster_size']:.1f}, "
              f"Min: {cluster_info['min_cluster_size']}, "
              f"Max: {cluster_info['max_cluster_size']}")
    
    def _assign_children_via_hnsw(self):
        """Assign children to each centroid using HNSW search."""
        print(f"Assigning children via HNSW (ef={self.child_search_ef})...")
        
        assignment_counts = defaultdict(int) if self.diversify_max_assignments else None
        
        for i, centroid_id in enumerate(self.centroid_ids):
            centroid_vector = self.centroids[i]
            
            # Find nearest neighbors using HNSW
            neighbors = self.base_index.query(
                centroid_vector, 
                k=self.k_children,
                ef=self.child_search_ef
            )
            
            # Extract children and apply diversification if enabled
            children = []
            for node_id, distance in neighbors:
                if self.diversify_max_assignments is None:
                    children.append(node_id)
                    self.child_vectors[node_id] = self.base_index[node_id]
                else:
                    # Diversification: limit assignments per child
                    if assignment_counts[node_id] < self.diversify_max_assignments:
                        children.append(node_id)
                        assignment_counts[node_id] += 1
                        self.child_vectors[node_id] = self.base_index[node_id]
            
            self.parent_child_map[centroid_id] = children
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processed {i + 1}/{self.n_clusters} centroids, "
                      f"found {len(children)} children for centroid {i}")
        
        # Repair phase: ensure minimum assignments
        if self.repair_min_assignments and assignment_counts is not None:
            self._repair_child_assignments(assignment_counts)
    
    def _repair_child_assignments(self, assignment_counts: Dict[Hashable, int]):
        """Repair phase: ensure every child has minimum assignments."""
        print(f"Repair phase: ensuring minimum {self.repair_min_assignments} assignments...")
        
        # Find under-assigned children
        all_base_nodes = set(self.base_index.keys())
        assigned_nodes = set(assignment_counts.keys())
        unassigned_nodes = all_base_nodes - assigned_nodes
        
        under_assigned = {
            node_id for node_id, count in assignment_counts.items()
            if count < self.repair_min_assignments
        }
        
        # Include completely unassigned nodes
        under_assigned.update(unassigned_nodes)
        
        print(f"Found {len(under_assigned)} under-assigned nodes "
              f"({len(unassigned_nodes)} completely unassigned)")
        
        # For each under-assigned node, find closest centroids
        for node_id in under_assigned:
            current_assignments = assignment_counts.get(node_id, 0)
            needed_assignments = self.repair_min_assignments - current_assignments
            
            if needed_assignments <= 0:
                continue
            
            node_vector = self.base_index[node_id]
            
            # Find distances to all centroids
            centroid_distances = []
            for i, centroid_vector in enumerate(self.centroids):
                distance = self.distance_func(node_vector, centroid_vector)
                centroid_distances.append((distance, self.centroid_ids[i]))
            
            # Sort by distance and assign to closest centroids
            centroid_distances.sort()
            for j in range(min(needed_assignments, len(centroid_distances))):
                _, centroid_id = centroid_distances[j]
                if node_id not in self.parent_child_map[centroid_id]:
                    self.parent_child_map[centroid_id].append(node_id)
                    self.child_vectors[node_id] = self.base_index[node_id]
                    assignment_counts[node_id] += 1
        
        # Report coverage after repair
        final_assigned = set(assignment_counts.keys())
        coverage = len(final_assigned) / len(all_base_nodes)
        print(f"Repair completed. Final coverage: {coverage:.3f} "
              f"({len(final_assigned)}/{len(all_base_nodes)} nodes)")
    
    def _build_centroid_index(self):
        """Build vectorized centroid matrix for fast Stage 1 search."""
        if self.centroids is None:
            raise ValueError("Centroids not computed yet")
        
        self._centroid_matrix = self.centroids.copy()
        self._centroid_id_array = np.array(self.centroid_ids)
        
        print(f"Built centroid index with shape {self._centroid_matrix.shape}")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10, 
        n_probe: int = 10
    ) -> List[Tuple[Hashable, float]]:
        """
        Perform two-stage search: K-Means centroids → HNSW children.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            n_probe: Number of centroids to probe in Stage 1
            
        Returns:
            List of (node_id, distance) tuples, sorted by distance
        """
        if self.centroids is None:
            raise ValueError("K-Means HNSW system not built yet")
        
        start = time.time()
        
        # Stage 1: Find closest centroids
        closest_centroids = self._stage1_centroid_search(query_vector, n_probe)
        
        # Stage 2: Search within children of selected centroids
        results = self._stage2_child_search(query_vector, closest_centroids, k)
        
        # Record timing
        elapsed = (time.time() - start) * 1000.0
        self.search_times.append(elapsed)
        self.stats['avg_search_time_ms'] = float(np.mean(self.search_times))
        
        return results
    
    def _stage1_centroid_search(
        self, 
        query_vector: np.ndarray, 
        n_probe: int
    ) -> List[Tuple[str, float]]:
        """Stage 1: Find closest K-Means centroids."""
        if self._centroid_matrix is not None:
            # Vectorized computation
            diffs = self._centroid_matrix - query_vector
            distances = np.linalg.norm(diffs, axis=1)
            indices = np.argsort(distances)[:n_probe]
            return [(self.centroid_ids[i], distances[i]) for i in indices]
        else:
            # Fallback loop-based computation
            centroid_distances = []
            for i, centroid_vector in enumerate(self.centroids):
                distance = self.distance_func(query_vector, centroid_vector)
                centroid_distances.append((distance, self.centroid_ids[i]))
            
            centroid_distances.sort()
            return [(cid, dist) for dist, cid in centroid_distances[:n_probe]]
    
    def _stage2_child_search(
        self, 
        query_vector: np.ndarray, 
        closest_centroids: List[Tuple[str, float]], 
        k: int
    ) -> List[Tuple[Hashable, float]]:
        """Stage 2: Search within children of selected centroids."""
        # Collect all candidate children
        candidate_children = set()
        for centroid_id, distance in closest_centroids:  # Fixed order: centroid_id first
            children = self.parent_child_map.get(centroid_id, [])
            candidate_children.update(children)
        
        # Optionally include centroids in results
        if self.include_centroids_in_results:
            for centroid_id, distance in closest_centroids:  # Fixed order: centroid_id first
                # Add centroid as a candidate (using centroid vector)
                centroid_idx = self.centroid_ids.index(centroid_id)
                centroid_vector = self.centroids[centroid_idx]
                # Store centroid vector temporarily
                self.child_vectors[centroid_id] = centroid_vector
                candidate_children.add(centroid_id)
        
        if not candidate_children:
            return []
        
        # Compute distances to all candidates
        candidate_ids = list(candidate_children)
        vectors = []
        valid_ids = []
        
        for cid in candidate_ids:
            if cid in self.child_vectors:
                vectors.append(self.child_vectors[cid])
                valid_ids.append(cid)
        
        if not vectors:
            return []
        
        # Vectorized distance computation
        candidate_matrix = np.vstack(vectors)
        diffs = candidate_matrix - query_vector
        distances = np.linalg.norm(diffs, axis=1)
        
        # Get top-k results
        indices = np.argsort(distances)[:k]
        results = [(valid_ids[i], distances[i]) for i in indices]
        
        # Record candidate size for statistics
        self.candidate_sizes.append(len(valid_ids))
        self.stats['avg_candidate_size'] = float(np.mean(self.candidate_sizes))
        
        return results
    
    def update_search_params(
        self, 
        n_clusters: Optional[int] = None,
        k_children: Optional[int] = None,
        child_search_ef: Optional[int] = None,
        rebuild: bool = False
    ):
        """
        Update search parameters and optionally rebuild the system.
        
        Args:
            n_clusters: New number of clusters
            k_children: New number of children per cluster
            child_search_ef: New search width for child finding
            rebuild: Whether to rebuild the system with new parameters
        """
        old_params = {
            'n_clusters': self.n_clusters,
            'k_children': self.k_children,
            'child_search_ef': self.child_search_ef
        }
        
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if k_children is not None:
            self.k_children = k_children
        if child_search_ef is not None:
            self.child_search_ef = child_search_ef
        
        print(f"Updated parameters: {old_params} -> "
              f"{{'n_clusters': {self.n_clusters}, 'k_children': {self.k_children}, "
              f"'child_search_ef': {self.child_search_ef}}}")
        
        if rebuild:
            print("Rebuilding K-Means HNSW system with new parameters...")
            self._build_kmeans_hnsw_system()
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the system."""
        stats = self.stats.copy()
        stats.update(self._overlap_stats)
        
        # Add K-Means specific stats
        if self.kmeans_model:
            cluster_info = self.kmeans_model.get_cluster_info()
            stats.update({
                'kmeans_inertia': cluster_info['inertia'],
                'kmeans_iterations': cluster_info['n_iterations'],
                'cluster_size_stats': {
                    'avg': cluster_info['avg_cluster_size'],
                    'std': cluster_info['std_cluster_size'],
                    'min': cluster_info['min_cluster_size'],
                    'max': cluster_info['max_cluster_size']
                }
            })
        
        return stats
    
    def get_centroid_info(self) -> Dict:
        """Get detailed information about centroids and mappings."""
        return {
            'centroid_child_map': self.parent_child_map,
            'num_clusters': self.n_clusters,
            'num_children': len(self.child_vectors),
            'k_children': self.k_children,
            'centroids_shape': self.centroids.shape if self.centroids is not None else None,
            'centroid_ids': self.centroid_ids
        }
    
    def _compute_mapping_diagnostics(self):
        """Compute coverage and overlap statistics for the mapping."""
        if not self.parent_child_map:
            self.stats['coverage_fraction'] = 0.0
            return
        
        all_children_sets = [set(children) for children in self.parent_child_map.values() if children]
        if not all_children_sets:
            self.stats['coverage_fraction'] = 0.0
            return
        
        # Coverage computation
        union_all = set().union(*all_children_sets)
        total_base_nodes = len(self.base_index)
        coverage_fraction = len(union_all) / total_base_nodes if total_base_nodes > 0 else 0.0
        self.stats['coverage_fraction'] = coverage_fraction
        
        # Sample pairwise overlaps for efficiency
        overlaps = []
        if len(all_children_sets) > 1:
            import random
            sample_pairs = min(self.overlap_sample, len(all_children_sets) * (len(all_children_sets) - 1) // 2)
            sampled_indices = random.sample(range(len(all_children_sets)), min(len(all_children_sets), 2 * int(np.sqrt(sample_pairs))))
            
            for i in range(len(sampled_indices)):
                for j in range(i + 1, len(sampled_indices)):
                    set_i = all_children_sets[sampled_indices[i]]
                    set_j = all_children_sets[sampled_indices[j]]
                    if set_i and set_j:  # Avoid empty sets
                        jaccard = len(set_i & set_j) / len(set_i | set_j)
                        overlaps.append(jaccard)
        
        if overlaps:
            self._overlap_stats = {
                'avg_jaccard_overlap': float(np.mean(overlaps)),
                'std_jaccard_overlap': float(np.std(overlaps)),
                'max_jaccard_overlap': float(np.max(overlaps)),
                'overlap_samples': len(overlaps)
            }
        else:
            self._overlap_stats = {
                'avg_jaccard_overlap': 0.0,
                'std_jaccard_overlap': 0.0,
                'max_jaccard_overlap': 0.0,
                'overlap_samples': 0
            }


def create_synthetic_dataset(n_vectors: int, dim: int, seed: int = 42) -> np.ndarray:
    """Create a synthetic dataset for testing K-Means HNSW."""
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
    # Example usage and testing
    print("K-Means HNSW System - Example Usage")
    
    # Create synthetic data
    print("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(5000, 128)  # 5K vectors for demo
    query_vectors, query_ids = create_query_set(dataset, 50)  # 50 queries
    
    # Build base HNSW index (Phase 1)
    print("Building base HNSW index...")
    distance_func = lambda x, y: np.linalg.norm(x - y)
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
    
    # Insert all vectors except queries
    for i, vector in enumerate(dataset):
        if i not in query_ids:
            base_index.insert(i, vector)
    
    print(f"Base HNSW index built with {len(base_index)} vectors")
    
    # Build K-Means HNSW system (Phase 2)
    print("Building K-Means HNSW system...")
    kmeans_hnsw = KMeansHNSW(
        base_index=base_index,
        n_clusters=50,  # 50 clusters for 5K dataset
        k_children=500,
        kmeans_params={'verbose': True}
    )
    
    # Test search (Phase 3)
    print("Testing search...")
    query_vector = query_vectors[0]
    results = kmeans_hnsw.search(query_vector, k=10, n_probe=5)
    
    print(f"\nSearch results for query 0:")
    for i, (node_id, distance) in enumerate(results):
        print(f"  {i+1}. Node {node_id}: distance = {distance:.4f}")
    
    # Print system statistics
    stats = kmeans_hnsw.get_stats()
    print(f"\nSystem Statistics:")
    print(f"  Construction time: {stats['total_construction_time']:.2f}s")
    print(f"  K-Means time: {stats['kmeans_fit_time']:.2f}s") 
    print(f"  Child mapping time: {stats['child_mapping_time']:.2f}s")
    print(f"  Coverage: {stats['coverage_fraction']:.3f}")
    print(f"  Avg children per cluster: {stats['avg_children_per_centroid']:.1f}")
    print(f"  Avg search time: {stats['avg_search_time_ms']:.2f}ms")
