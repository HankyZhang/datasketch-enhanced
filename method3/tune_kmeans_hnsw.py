"""
Method 3 Parameter Tuning: K-Means HNSW System

This module provides parameter tuning and optimization for the K-Means HNSW system.
It includes comprehensive evaluation, parameter sweeps, and performance analysis.
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from itertools import product

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from method3.kmeans_hnsw import KMeansHNSW
from hnsw.hnsw import HNSW
# Switch to sklearn MiniBatchKMeans for pure k-means baseline
from sklearn.cluster import MiniBatchKMeans


class KMeansHNSWEvaluator:
    """
    Comprehensive evaluator for K-Means HNSW system performance.
    Compatible with existing evaluation frameworks from Methods 1 & 2.
    """
    
    def __init__(
        self, 
        dataset: np.ndarray, 
        query_set: np.ndarray, 
        query_ids: List[int],
        distance_func: callable
    ):
        """
        Initialize the evaluator.
        
        Args:
            dataset: Full dataset vectors (shape: [n_vectors, dim])
            query_set: Query vectors (shape: [n_queries, dim])
            query_ids: IDs for query vectors
            distance_func: Distance function for ground truth computation
        """
        self.dataset = dataset
        self.query_set = query_set
        self.query_ids = query_ids
        self.distance_func = distance_func
        
        # Ground truth cache
        self._ground_truth_cache = {}
    
    def compute_ground_truth(
        self, 
        k: int, 
        exclude_query_ids: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute ground truth using brute force search.
        
        Args:
            k: Number of nearest neighbors
            exclude_query_ids: Whether to exclude query IDs from results
            
        Returns:
            Dictionary mapping query_id to list of (neighbor_id, distance) tuples
        """
        cache_key = (k, exclude_query_ids)
        if cache_key in self._ground_truth_cache:
            return self._ground_truth_cache[cache_key]
        
        print(f"Computing ground truth for {len(self.query_set)} queries (k={k})...")
        start_time = time.time()
        
        ground_truth = {}
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            distances = []
            
            for j, data_vector in enumerate(self.dataset):
                if exclude_query_ids and j == query_id:
                    continue  # Skip the query itself
                
                distance = self.distance_func(query_vector, data_vector)
                distances.append((distance, j))
            
            # Sort by distance and take top-k
            distances.sort()
            ground_truth[query_id] = distances[:k]
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(self.query_set)} queries")
        
        elapsed = time.time() - start_time
        print(f"Ground truth computed in {elapsed:.2f}s")
        
        self._ground_truth_cache[cache_key] = ground_truth
        return ground_truth
    
    def evaluate_recall(
        self,
        kmeans_hnsw: KMeansHNSW,
        k: int,
        n_probe: int,
        ground_truth: Optional[Dict] = None,
        exclude_query_ids: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate recall performance of the K-Means HNSW system.
        
        Args:
            kmeans_hnsw: The K-Means HNSW system to evaluate
            k: Number of results to return
            n_probe: Number of centroids to probe
            ground_truth: Precomputed ground truth (optional)
            exclude_query_ids: Whether to exclude query IDs from evaluation
            
        Returns:
            Dictionary containing recall metrics and performance data
        """
        if ground_truth is None:
            ground_truth = self.compute_ground_truth(k, exclude_query_ids)
        
        print(f"Evaluating recall for {len(self.query_set)} queries (k={k}, n_probe={n_probe})...")
        start_time = time.time()
        
        total_correct = 0
        total_expected = len(self.query_set) * k
        query_times = []
        individual_recalls = []
        
        for i, (query_vector, query_id) in enumerate(zip(self.query_set, self.query_ids)):
            # Get ground truth for this query
            true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
            
            # Search using K-Means HNSW
            search_start = time.time()
            results = kmeans_hnsw.search(query_vector, k=k, n_probe=n_probe)
            search_time = time.time() - search_start
            query_times.append(search_time)
            
            # Count correct results
            found_neighbors = {node_id for node_id, _ in results}
            correct = len(true_neighbors & found_neighbors)
            total_correct += correct
            
            # Individual recall for this query
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)
            
            if (i + 1) % 20 == 0:
                current_recall = total_correct / ((i + 1) * k)
                print(f"  Processed {i + 1}/{len(self.query_set)} queries, "
                      f"current recall: {current_recall:.4f}")
        
        # Calculate final metrics
        overall_recall = total_correct / total_expected
        avg_query_time = np.mean(query_times)
        std_query_time = np.std(query_times)
        total_evaluation_time = time.time() - start_time
        
        return {
            'recall_at_k': overall_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': np.mean(individual_recalls),
            'std_individual_recall': np.std(individual_recalls),
            'avg_query_time_ms': avg_query_time * 1000,
            'std_query_time_ms': std_query_time * 1000,
            'total_evaluation_time': total_evaluation_time,
            'k': k,
            'n_probe': n_probe,
            'system_stats': kmeans_hnsw.get_stats()
        }

    # -------------------- Phase-Specific Evaluations --------------------
    def evaluate_hnsw_baseline(
        self,
        base_index: HNSW,
        k: int,
        ef: int,
        ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate recall using ONLY the base HNSW index (Phase 1)."""
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        for query_vector, query_id in zip(self.query_set, self.query_ids):
            true_neighbors = {nid for _, nid in ground_truth[query_id]}
            t0 = time.time()
            results = base_index.query(query_vector, k=k, ef=ef)
            dt = time.time() - t0
            query_times.append(dt)
            found = {nid for nid, _ in results}
            total_correct += len(true_neighbors & found)
        return {
            'phase': 'baseline_hnsw',
            'ef': ef,
            'recall_at_k': total_correct / total_expected if total_expected else 0.0,
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'total_correct': total_correct,
            'total_expected': total_expected
        }

    def evaluate_clusters_only(
        self,
        kmeans_model: Any,
        dataset: np.ndarray,
        k: int,
        ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate recall using ONLY KMeans clusters (Phase 2) without child mapping.
        Strategy: For each query pick nearest centroid, restrict to its members, pick top-k by L2.
        """
        if not hasattr(kmeans_model, 'cluster_centers_') or not hasattr(kmeans_model, 'labels_'):
            raise ValueError("KMeans model must be fitted with cluster_centers_ and labels_ available")
        centers = kmeans_model.cluster_centers_
        labels = kmeans_model.labels_
        n_clusters = centers.shape[0]
        # Build inverse index: cluster -> indices
        clusters = [[] for _ in range(n_clusters)]
        for idx, c in enumerate(labels):
            clusters[c].append(idx)
        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []
        for qvec, qid in zip(self.query_set, self.query_ids):
            t0 = time.time()
            d2c = np.linalg.norm(centers - qvec, axis=1)
            cidx = int(np.argmin(d2c))
            member_ids = clusters[cidx]
            if member_ids:
                member_vecs = dataset[member_ids]
                dists = np.linalg.norm(member_vecs - qvec, axis=1)
                # exclude identical id if present
                pairs = [(float(dist), int(mid)) for dist, mid in zip(dists, member_ids) if mid != qid]
                pairs.sort(key=lambda x: x[0])
                results = pairs[:k]
                found = {mid for _, mid in results}
            else:
                found = set()
            query_times.append(time.time() - t0)
            true_neighbors = {nid for _, nid in ground_truth[qid]}
            correct = len(true_neighbors & found)
            total_correct += correct
            individual_recalls.append(correct / k if k else 0.0)
        overall = total_correct / total_expected if total_expected else 0.0
        return {
            'phase': 'clusters_only',
            'recall_at_k': overall,
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'total_correct': total_correct,
            'total_expected': total_expected,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'n_clusters': n_clusters
        }
    
    def parameter_sweep(
        self,
        base_index: HNSW,
        param_grid: Dict[str, List[Any]],
        evaluation_params: Dict[str, Any],
        max_combinations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform comprehensive parameter sweep for optimization.
        
        Args:
            base_index: Base HNSW index to use
            param_grid: Dictionary of parameters and their values to test
            evaluation_params: Parameters for evaluation (k, n_probe_values)
            max_combinations: Maximum number of combinations to test
            
        Returns:
            List of evaluation results for each parameter combination
        """
        print("Starting parameter sweep for K-Means HNSW...")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        if max_combinations and len(combinations) > max_combinations:
            print(f"Limiting to {max_combinations} combinations out of {len(combinations)}")
            import random
            combinations = random.sample(combinations, max_combinations)
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        k_values = evaluation_params.get('k_values', [10])
        n_probe_values = evaluation_params.get('n_probe_values', [5, 10, 20])
        
        # Precompute ground truth for all k values
        ground_truths = {}
        for k in k_values:
            ground_truths[k] = self.compute_ground_truth(k)

        for i, combination in enumerate(combinations):
            print(f"\n--- Combination {i + 1}/{len(combinations)} ---")

            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            print(f"Parameters: {params}")

            try:
                phase_records = []
                # Phase 1: baseline HNSW recall (optional multiple ef values)
                baseline_ef_values = evaluation_params.get('baseline_ef_values', [evaluation_params.get('baseline_ef', 100)])
                for k in k_values:
                    for ef in baseline_ef_values:
                        b_eval = self.evaluate_hnsw_baseline(base_index, k, ef, ground_truths[k])
                        phase_records.append({**b_eval, 'k': k})

                # Build full system (includes clustering + child mapping)
                construction_start = time.time()
                kmeans_hnsw = KMeansHNSW(base_index=base_index, **params)
                construction_time = time.time() - construction_start

                # Phase 2: clusters-only recall using fitted internal KMeans model
                for k in k_values:
                    c_eval = self.evaluate_clusters_only(
                        kmeans_hnsw.kmeans_model,
                        kmeans_hnsw._extract_dataset_vectors(),  # reuse extractor
                        k,
                        ground_truths[k]
                    )
                    phase_records.append({**c_eval, 'k': k})

                # Phase 3: full two-stage search evaluations over n_probe
                for k in k_values:
                    for n_probe in n_probe_values:
                        eval_result = self.evaluate_recall(kmeans_hnsw, k, n_probe, ground_truths[k])
                        phase_records.append({**eval_result, 'phase': 'kmeans_hnsw_two_stage'})

                combination_results = {
                    'parameters': params,
                    'construction_time': construction_time,
                    'phase_evaluations': phase_records
                }
                results.append(combination_results)
                best_recall = max(r['recall_at_k'] for r in phase_records if 'recall_at_k' in r)
                print(f"Best recall (any phase) for this combination: {best_recall:.4f}")
            except Exception as e:
                print(f"Error with combination {params}: {e}")
                continue
        
        print(f"\nParameter sweep completed. Tested {len(results)} combinations.")
        return results
    
    def find_optimal_parameters(
        self,
        sweep_results: List[Dict[str, Any]],
        optimization_target: str = 'recall_at_k',
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal parameters from sweep results.
        
        Args:
            sweep_results: Results from parameter_sweep()
            optimization_target: Metric to optimize ('recall_at_k', 'avg_query_time_ms', etc.)
            constraints: Constraints on other metrics (e.g., {'avg_query_time_ms': 50.0})
            
        Returns:
            Dictionary containing optimal parameters and their performance
        """
        print(f"Finding optimal parameters optimizing for {optimization_target}...")
        
        best_result = None
        best_value = -float('inf') if 'recall' in optimization_target else float('inf')
        
        for result in sweep_results:
            for evaluation in result.get('phase_evaluations', []):
                # Check constraints
                if constraints:
                    violates_constraint = False
                    for constraint_metric, constraint_value in constraints.items():
                        if constraint_metric in evaluation:
                            if evaluation[constraint_metric] > constraint_value:
                                violates_constraint = True
                                break
                    if violates_constraint:
                        continue
                
                # Check if this is better
                current_value = evaluation.get(optimization_target)
                if current_value is None:
                    continue
                
                is_better = (
                    (current_value > best_value and 'recall' in optimization_target) or
                    (current_value < best_value and 'time' in optimization_target)
                )
                
                if is_better:
                    best_value = current_value
                    best_result = {
                        'parameters': result['parameters'],
                        'performance': evaluation,
                        'construction_time': result['construction_time']
                    }
        
        if best_result:
            print(f"Optimal parameters found:")
            print(f"  Parameters: {best_result['parameters']}")
            print(f"  {optimization_target}: {best_value:.4f}")
            print(f"  Construction time: {best_result['construction_time']:.2f}s")
        else:
            print("No valid parameters found satisfying constraints.")
        
        return best_result or {}
    
    def compare_with_baselines(
        self,
        kmeans_hnsw: KMeansHNSW,
        base_index: HNSW,
        k: int = 10,
        n_probe: int = 10,
        ef_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Compare K-Means HNSW performance with baseline HNSW and pure K-means.
        
        Args:
            kmeans_hnsw: K-Means HNSW system
            base_index: Baseline HNSW index
            k: Number of results
            n_probe: Number of centroids to probe for K-Means HNSW
            ef_values: List of ef values to test for baseline HNSW
            
        Returns:
            Comparison results including pure K-means clustering
        """
        if ef_values is None:
            ef_values = [50, 100, 200, 400]
        
        print(f"Comparing K-Means HNSW with baseline HNSW and pure K-means...")
        
        ground_truth = self.compute_ground_truth(k)
        
        # Evaluate K-Means HNSW
        kmeans_result = self.evaluate_recall(kmeans_hnsw, k, n_probe, ground_truth)
        
        # Evaluate pure K-means clustering
        print("Evaluating pure K-means clustering...")
        kmeans_clustering_result = self._evaluate_pure_kmeans(k, ground_truth)
        
        # Evaluate baseline HNSW with different ef values
        baseline_results = []
        for ef in ef_values:
            print(f"Evaluating baseline HNSW with ef={ef}...")
            
            query_times = []
            total_correct = 0
            total_expected = len(self.query_set) * k
            
            for query_vector, query_id in zip(self.query_set, self.query_ids):
                true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
                
                search_start = time.time()
                results = base_index.query(query_vector, k=k, ef=ef)
                search_time = time.time() - search_start
                query_times.append(search_time)
                
                found_neighbors = {node_id for node_id, _ in results}
                total_correct += len(true_neighbors & found_neighbors)
            
            baseline_result = {
                'method': 'baseline_hnsw',
                'ef': ef,
                'recall_at_k': total_correct / total_expected,
                'avg_query_time_ms': np.mean(query_times) * 1000,
                'total_correct': total_correct,
                'total_expected': total_expected
            }
            baseline_results.append(baseline_result)
        
        return {
            'kmeans_hnsw': kmeans_result,
            'pure_kmeans': kmeans_clustering_result,
            'baseline_hnsw': baseline_results,
            'comparison_summary': {
                'kmeans_hnsw_recall': kmeans_result['recall_at_k'],
                'kmeans_hnsw_time_ms': kmeans_result['avg_query_time_ms'],
                'pure_kmeans_recall': kmeans_clustering_result['recall_at_k'],
                'pure_kmeans_time_ms': kmeans_clustering_result['avg_query_time_ms'],
                'best_baseline_recall': max(r['recall_at_k'] for r in baseline_results),
                'best_baseline_time_ms': min(r['avg_query_time_ms'] for r in baseline_results)
            }
        }
    
    def _evaluate_pure_kmeans(self, k: int, ground_truth: Dict) -> Dict[str, Any]:
        """Evaluate pure MiniBatchKMeans clustering for comparison.
        Uses centroids to pick a single cluster, then returns top-k points within that cluster.
        """
        print("Running MiniBatchKMeans clustering (baseline)...")

        # Heuristic: choose number of clusters proportional to sqrt(N) if feasible, fallback to 10
        n_samples = len(self.dataset)
        suggested = int(np.sqrt(n_samples))
        n_clusters = max(10, min(256, suggested))

        start_time = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1024, n_samples),
            n_init=3,
            max_iter=100,
            verbose=0
        )
        kmeans.fit(self.dataset)
        clustering_time = time.time() - start_time
        print(f"MiniBatchKMeans clustering completed in {clustering_time:.2f}s (n_clusters={n_clusters})")

        query_times = []
        total_correct = 0
        total_expected = len(self.query_set) * k
        individual_recalls = []

        for query_vector, query_id in zip(self.query_set, self.query_ids):
            search_start = time.time()
            # Vectorized distance to centroids
            diffs = kmeans.cluster_centers_ - query_vector
            distances_to_centroids = np.linalg.norm(diffs, axis=1)
            nearest_cluster = int(np.argmin(distances_to_centroids))

            cluster_points = np.where(kmeans.labels_ == nearest_cluster)[0]
            if len(cluster_points) > 0:
                cluster_vecs = self.dataset[cluster_points]
                point_diffs = cluster_vecs - query_vector
                dists = np.linalg.norm(point_diffs, axis=1)
                # Exclude the query id if present
                filtered = [
                    (dist, int(pid)) for dist, pid in zip(dists, cluster_points)
                    if pid != query_id
                ]
                filtered.sort(key=lambda x: x[0])
                results = filtered[:k]
                found_neighbors = {pid for _, pid in results}
            else:
                found_neighbors = set()

            search_time = time.time() - search_start
            query_times.append(search_time)

            true_neighbors = {neighbor_id for _, neighbor_id in ground_truth[query_id]}
            correct = len(true_neighbors & found_neighbors)
            total_correct += correct
            individual_recall = correct / k if k > 0 else 0.0
            individual_recalls.append(individual_recall)

        overall_recall = total_correct / total_expected if total_expected > 0 else 0.0

        return {
            'method': 'pure_minibatch_kmeans',
            'recall_at_k': overall_recall,
            'total_correct': total_correct,
            'total_expected': total_expected,
            'individual_recalls': individual_recalls,
            'avg_individual_recall': float(np.mean(individual_recalls)),
            'std_individual_recall': float(np.std(individual_recalls)),
            'avg_query_time_ms': float(np.mean(query_times) * 1000),
            'std_query_time_ms': float(np.std(query_times) * 1000),
            'clustering_time': clustering_time,
            'n_clusters': n_clusters,
            'k': k
        }


def save_results(results: Dict[str, Any], filename: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {filename}")


def load_sift_data():
    """Load SIFT dataset for evaluation."""
    sift_dir = os.path.join(os.path.dirname(__file__), '..', 'sift')
    
    try:
        def read_fvecs(path: str, max_vectors: Optional[int] = None) -> np.ndarray:
            """Read .fvecs file (FAISS format). Each vector stored as: int32 dim + dim float32.
            This implementation avoids mis-parsing by reading int32 header first.
            """
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            raw = np.fromfile(path, dtype=np.int32)
            if raw.size == 0:
                raise ValueError(f"Empty fvecs file: {path}")
            dim = raw[0]
            if dim <= 0 or dim > 4096:
                raise ValueError(f"Unreasonable vector dimension {dim} parsed from {path}")
            record_size = dim + 1
            count = raw.size // record_size
            raw = raw.reshape(count, record_size)
            vecs = raw[:, 1:].astype(np.float32)
            if max_vectors is not None and count > max_vectors:
                vecs = vecs[:max_vectors]
            return vecs

        base_path = os.path.join(sift_dir, 'sift_base.fvecs')
        query_path = os.path.join(sift_dir, 'sift_query.fvecs')

        # Limit for tuning demo to keep runtime reasonable
        base_vectors = read_fvecs(base_path, max_vectors=50000)
        query_vectors = read_fvecs(query_path, max_vectors=1000)

        print(f"Loaded SIFT data: {base_vectors.shape[0]} base vectors, "
              f"{query_vectors.shape[0]} query vectors, dimension {base_vectors.shape[1]}")

        return base_vectors, query_vectors
    
    except Exception as e:
        print(f"Error loading SIFT data: {e}")
        print("Using synthetic data instead...")
        return None, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="K-Means HNSW Parameter Tuning and Evaluation")
    parser.add_argument('--dataset-size', type=int, default=10000, help='Number of base vectors to use (default: 10000 ~1w)')
    parser.add_argument('--query-size', type=int, default=50, help='Number of query vectors to use (default: 50)')
    parser.add_argument('--dimension', type=int, default=128, help='Vector dimensionality for synthetic data (if SIFT not loaded)')
    parser.add_argument('--no-sift', action='store_true', help='Force synthetic data even if SIFT files exist')
    args = parser.parse_args()

    print("K-Means HNSW Parameter Tuning and Evaluation")
    print(f"Requested dataset size: {args.dataset_size}, query size: {args.query_size}")
    
    # Try to load SIFT data, fall back to synthetic unless disabled
    base_vectors, query_vectors = (None, None)
    if not args.no_sift:
        base_vectors, query_vectors = load_sift_data()
    
    if base_vectors is None:
        # Create synthetic data
        print("Creating synthetic dataset...")
        base_vectors = np.random.randn(max(args.dataset_size, 10000), args.dimension).astype(np.float32)
        query_vectors = np.random.randn(max(args.query_size, 100), args.dimension).astype(np.float32)
    
    # Slice to requested sizes (cap by available)
    if len(base_vectors) > args.dataset_size:
        base_vectors = base_vectors[:args.dataset_size]
    if len(query_vectors) > args.query_size:
        query_vectors = query_vectors[:args.query_size]
    print(f"Using base vectors: {len(base_vectors)} | queries: {len(query_vectors)} | dim: {base_vectors.shape[1]}")
    query_ids = list(range(len(query_vectors)))
    
    # Distance function
    distance_func = lambda x, y: np.linalg.norm(x - y)
    
    # Build base HNSW index
    print("Building base HNSW index...")
    base_index = HNSW(distance_func=distance_func, m=16, ef_construction=100)
    
    for i, vector in enumerate(base_vectors):
        base_index.insert(i, vector)
        if (i + 1) % 1000 == 0:
            print(f"  Inserted {i + 1}/{len(base_vectors)} vectors")
    
    print(f"Base HNSW index built with {len(base_index)} vectors")
    
    # Initialize evaluator
    evaluator = KMeansHNSWEvaluator(base_vectors, query_vectors, query_ids, distance_func)
    
    # Define parameter grid for sweep
    # Adjust default cluster count heuristics for larger datasets: scale choices
    if args.dataset_size <= 2000:
        cluster_options = [10]
    elif args.dataset_size <= 5000:
        cluster_options = [16, 32]
    else:
        cluster_options = [32, 64, 128]

    param_grid = {
        'n_clusters': cluster_options,
        'k_children': [200],
        'child_search_ef': [300]
    }
    
    evaluation_params = {
        'k_values': [10],
        'n_probe_values': [5, 10, 20]
    }
    
    # Perform parameter sweep
    print("\nStarting parameter sweep...")
    # Limit combinations to keep runtime sane on large sets
    max_combos = 9 if len(cluster_options) > 1 else None
    sweep_results = evaluator.parameter_sweep(
        base_index,
        param_grid,
        evaluation_params,
        max_combinations=max_combos
    )
    
    # Find optimal parameters
    optimal = evaluator.find_optimal_parameters(
        sweep_results,
        optimization_target='recall_at_k',
        constraints={'avg_query_time_ms': 100.0}  # Max 100ms per query
    )
    
    if optimal:
        # Build system with optimal parameters and compare with baseline
        print("\nBuilding system with optimal parameters...")
        optimal_kmeans_hnsw = KMeansHNSW(
            base_index=base_index,
            **optimal['parameters']
        )
        
        comparison = evaluator.compare_with_baselines(
            optimal_kmeans_hnsw,
            base_index,
            k=10,
            n_probe=10
        )
        
        print("\nComparison Results:")
        print(f"K-Means HNSW: Recall={comparison['comparison_summary']['kmeans_hnsw_recall']:.4f}, "
              f"Time={comparison['comparison_summary']['kmeans_hnsw_time_ms']:.2f}ms")
        print(f"Pure K-Means: Recall={comparison['comparison_summary']['pure_kmeans_recall']:.4f}, "
              f"Time={comparison['comparison_summary']['pure_kmeans_time_ms']:.2f}ms")
        print(f"Best Baseline: Recall={comparison['comparison_summary']['best_baseline_recall']:.4f}, "
              f"Time={comparison['comparison_summary']['best_baseline_time_ms']:.2f}ms")
        
        # Additional detailed output for pure K-means
        pure_kmeans_result = comparison['pure_kmeans']
        print(f"\nDetailed Pure K-Means Results:")
        print(f"  Overall Recall@{pure_kmeans_result['k']}: {pure_kmeans_result['recall_at_k']:.4f}")
        print(f"  Average Individual Recall: {pure_kmeans_result['avg_individual_recall']:.4f}")
        print(f"  Correct/Expected: {pure_kmeans_result['total_correct']}/{pure_kmeans_result['total_expected']}")
        print(f"  Clustering Time: {pure_kmeans_result['clustering_time']:.2f}s")
        print(f"  Average Query Time: {pure_kmeans_result['avg_query_time_ms']:.2f}ms")
        
        # Save results
        results = {
            'sweep_results': sweep_results,
            'optimal_parameters': optimal,
            'baseline_comparison': comparison,
            'evaluation_info': {
                'dataset_size': len(base_vectors),
                'query_size': len(query_vectors),
                'dimension': base_vectors.shape[1],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        save_results(results, 'method3_tuning_results.json')
        
    print("\nParameter tuning completed!")
