#!/usr/bin/env python3
"""
HNSW Hybrid System - Full Experimental Pipeline

This script implements the complete experimental pipeline as described in the project guide:
- Phase 1: Project objectives and core concept definition
- Phase 2: Data preparation and baseline construction  
- Phase 3: Building custom parent-child index structure
- Phase 4: Implementing two-stage search logic
- Phase 5: Experimental evaluation with recall@K metrics

Usage:
    python experiment_runner.py --dataset_size 6000000 --query_size 10000 --dim 128
"""

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

from datasketch.hnsw import HNSW
from hnsw_hybrid import HNSWHybrid, HNSWEvaluator, create_synthetic_dataset, create_query_set


class HNSWExperimentRunner:
    """
    Main experimental runner for the HNSW hybrid system evaluation.
    """
    
    def __init__(
        self,
        dataset_size: int = 6000000,
        query_size: int = 10000,
        dim: int = 128,
        parent_level: int = 2,
        k_children_values: List[int] = None,
        n_probe_values: List[int] = None,
        k_values: List[int] = None,
        output_dir: str = "experiment_results",
        seed: int = 42
    ):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_size: Size of the full dataset
            query_size: Size of the query set
            dim: Vector dimension
            parent_level: HNSW level to extract parent nodes from
            k_children_values: List of k_children values to test
            n_probe_values: List of n_probe values to test
            k_values: List of k values for recall evaluation
            output_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.dataset_size = dataset_size
        self.query_size = query_size
        self.dim = dim
        self.parent_level = parent_level
        self.k_children_values = k_children_values or [500, 1000, 2000]
        self.n_probe_values = n_probe_values or [5, 10, 20, 50]
        self.k_values = k_values or [10, 50, 100]
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data structures
        self.dataset: Optional[np.ndarray] = None
        self.query_vectors: Optional[np.ndarray] = None
        self.query_ids: Optional[List[int]] = None
        self.base_index: Optional[HNSW] = None
        self.evaluator: Optional[HNSWEvaluator] = None
        self.ground_truth: Optional[Dict] = None
        
        # Results storage
        self.results: Dict = {
            'experiment_config': {
                'dataset_size': dataset_size,
                'query_size': query_size,
                'dim': dim,
                'parent_level': parent_level,
                'k_children_values': self.k_children_values,
                'n_probe_values': self.n_probe_values,
                'k_values': self.k_values,
                'seed': seed
            },
            'phase_results': {}
        }
    
    def run_full_experiment(self):
        """Run the complete experimental pipeline."""
        print("=" * 80)
        print("HNSW HYBRID SYSTEM - FULL EXPERIMENTAL PIPELINE")
        print("=" * 80)
        
        try:
            # Phase 2: Data Preparation
            self._phase2_data_preparation()
            
            # Phase 3: Build Base HNSW Index
            self._phase3_build_base_index()
            
            # Phase 4: Build Hybrid Index
            self._phase4_build_hybrid_index()
            
            # Phase 5: Experimental Evaluation
            self._phase5_experimental_evaluation()
            
            # Save results
            self._save_results()
            
            # Generate analysis
            self._generate_analysis()
            
            print("\n" + "=" * 80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
        except Exception as e:
            print(f"\nERROR: Experiment failed with exception: {e}")
            raise
    
    def _phase2_data_preparation(self):
        """Phase 2: Data preparation and baseline construction."""
        print("\n" + "-" * 60)
        print("PHASE 2: DATA PREPARATION")
        print("-" * 60)
        
        start_time = time.time()
        
        # Create synthetic dataset
        print(f"Creating synthetic dataset: {self.dataset_size} vectors, dim={self.dim}")
        self.dataset = create_synthetic_dataset(self.dataset_size, self.dim, self.seed)
        
        # Create query set
        print(f"Creating query set: {self.query_size} queries")
        self.query_vectors, self.query_ids = create_query_set(
            self.dataset, self.query_size, self.seed + 1
        )
        
        # Initialize evaluator
        self.evaluator = HNSWEvaluator(self.dataset, self.query_vectors, self.query_ids)
        
        phase_time = time.time() - start_time
        self.results['phase_results']['phase2_data_preparation'] = {
            'dataset_shape': self.dataset.shape,
            'query_shape': self.query_vectors.shape,
            'phase_time': phase_time
        }
        
        print(f"Data preparation completed in {phase_time:.2f}s")
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"Query set shape: {self.query_vectors.shape}")
    
    def _phase3_build_base_index(self):
        """Phase 3: Build the base HNSW index."""
        print("\n" + "-" * 60)
        print("PHASE 3: BUILDING BASE HNSW INDEX")
        print("-" * 60)
        
        start_time = time.time()
        
        # Define distance function
        distance_func = lambda x, y: np.linalg.norm(x - y)
        
        # Build base HNSW index
        print("Building base HNSW index...")
        self.base_index = HNSW(
            distance_func=distance_func,
            m=16,
            ef_construction=200
        )
        
        # Insert all vectors except queries
        print(f"Inserting {self.dataset_size - self.query_size} vectors into base index...")
        for i, vector in enumerate(self.dataset):
            if i not in self.query_ids:  # Exclude query vectors
                self.base_index.insert(i, vector)
        
        phase_time = time.time() - start_time
        self.results['phase_results']['phase3_base_index'] = {
            'index_size': len(self.base_index),
            'num_layers': len(self.base_index._graphs),
            'phase_time': phase_time
        }
        
        print(f"Base index built in {phase_time:.2f}s")
        print(f"Index size: {len(self.base_index)} vectors")
        print(f"Number of layers: {len(self.base_index._graphs)}")
    
    def _phase4_build_hybrid_index(self):
        """Phase 4: Build hybrid parent-child index structure."""
        print("\n" + "-" * 60)
        print("PHASE 4: BUILDING HYBRID INDEX STRUCTURE")
        print("-" * 60)
        
        # Test different k_children values
        hybrid_indices = {}
        
        for k_children in self.k_children_values:
            print(f"\nBuilding hybrid index with k_children={k_children}")
            start_time = time.time()
            
            hybrid_index = HNSWHybrid(
                base_index=self.base_index,
                parent_level=self.parent_level,
                k_children=k_children,
                distance_func=self.base_index._distance_func
            )
            
            build_time = time.time() - start_time
            hybrid_indices[k_children] = hybrid_index
            
            print(f"Hybrid index built in {build_time:.2f}s")
            print(f"Stats: {hybrid_index.get_stats()}")
        
        self.hybrid_indices = hybrid_indices
        
        self.results['phase_results']['phase4_hybrid_index'] = {
            'hybrid_indices': {
                k: idx.get_stats() for k, idx in hybrid_indices.items()
            }
        }
    
    def _phase5_experimental_evaluation(self):
        """Phase 5: Experimental evaluation with recall@K metrics."""
        print("\n" + "-" * 60)
        print("PHASE 5: EXPERIMENTAL EVALUATION")
        print("-" * 60)
        
        # Compute ground truth for all k values
        print("Computing ground truth...")
        ground_truth_results = {}
        
        for k in self.k_values:
            print(f"Computing ground truth for k={k}")
            start_time = time.time()
            ground_truth = self.evaluator.compute_ground_truth(k, self.base_index._distance_func)
            compute_time = time.time() - start_time
            
            ground_truth_results[k] = {
                'ground_truth': ground_truth,
                'compute_time': compute_time
            }
        
        self.ground_truth_results = ground_truth_results
        
        # Run parameter sweep for each hybrid index
        all_results = []
        
        for k_children, hybrid_index in self.hybrid_indices.items():
            print(f"\nEvaluating hybrid index with k_children={k_children}")
            
            for k in self.k_values:
                print(f"  Evaluating with k={k}")
                ground_truth = ground_truth_results[k]['ground_truth']
                
                for n_probe in self.n_probe_values:
                    print(f"    Testing n_probe={n_probe}")
                    
                    result = self.evaluator.evaluate_recall(
                        hybrid_index, k=k, n_probe=n_probe, ground_truth=ground_truth
                    )
                    
                    # Add configuration info
                    result['k_children'] = k_children
                    result['configuration'] = f"k_children={k_children}, n_probe={n_probe}, k={k}"
                    
                    all_results.append(result)
                    
                    print(f"      Recall@{k}: {result['recall_at_k']:.4f}, "
                          f"Query time: {result['avg_query_time_ms']:.2f}ms")
        
        self.results['phase_results']['phase5_evaluation'] = {
            'ground_truth_computation': {
                k: {'compute_time': v['compute_time']} 
                for k, v in ground_truth_results.items()
            },
            'parameter_sweep_results': all_results
        }
    
    def _save_results(self):
        """Save experimental results to files."""
        print("\n" + "-" * 60)
        print("SAVING RESULTS")
        print("-" * 60)
        
        # Save main results
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_numpy_types(self.results)
            json.dump(json_results, f, indent=2)
        
        # Save detailed parameter sweep results
        sweep_file = self.output_dir / "parameter_sweep_results.json"
        sweep_results = self.results['phase_results']['phase5_evaluation']['parameter_sweep_results']
        with open(sweep_file, 'w') as f:
            json.dump(self._convert_numpy_types(sweep_results), f, indent=2)
        
        print(f"Results saved to {results_file}")
        print(f"Parameter sweep results saved to {sweep_file}")
    
    def _generate_analysis(self):
        """Generate analysis plots and summary."""
        print("\n" + "-" * 60)
        print("GENERATING ANALYSIS")
        print("-" * 60)
        
        sweep_results = self.results['phase_results']['phase5_evaluation']['parameter_sweep_results']
        
        # Create analysis plots
        self._plot_recall_vs_n_probe(sweep_results)
        self._plot_recall_vs_k_children(sweep_results)
        self._plot_query_time_vs_recall(sweep_results)
        self._generate_summary_table(sweep_results)
        
        print("Analysis plots and summary generated")
    
    def _plot_recall_vs_n_probe(self, results: List[Dict]):
        """Plot recall vs n_probe for different k_children values."""
        fig, axes = plt.subplots(1, len(self.k_values), figsize=(5*len(self.k_values), 4))
        if len(self.k_values) == 1:
            axes = [axes]
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            for k_children in self.k_children_values:
                # Filter results for this k and k_children
                filtered_results = [
                    r for r in results 
                    if r['k'] == k and r['k_children'] == k_children
                ]
                
                if filtered_results:
                    n_probes = [r['n_probe'] for r in filtered_results]
                    recalls = [r['recall_at_k'] for r in filtered_results]
                    
                    ax.plot(n_probes, recalls, 'o-', label=f'k_children={k_children}')
            
            ax.set_xlabel('n_probe')
            ax.set_ylabel(f'Recall@{k}')
            ax.set_title(f'Recall@{k} vs n_probe')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recall_vs_n_probe.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_recall_vs_k_children(self, results: List[Dict]):
        """Plot recall vs k_children for different n_probe values."""
        fig, axes = plt.subplots(1, len(self.k_values), figsize=(5*len(self.k_values), 4))
        if len(self.k_values) == 1:
            axes = [axes]
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            for n_probe in self.n_probe_values:
                # Filter results for this k and n_probe
                filtered_results = [
                    r for r in results 
                    if r['k'] == k and r['n_probe'] == n_probe
                ]
                
                if filtered_results:
                    k_children_list = [r['k_children'] for r in filtered_results]
                    recalls = [r['recall_at_k'] for r in filtered_results]
                    
                    ax.plot(k_children_list, recalls, 'o-', label=f'n_probe={n_probe}')
            
            ax.set_xlabel('k_children')
            ax.set_ylabel(f'Recall@{k}')
            ax.set_title(f'Recall@{k} vs k_children')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recall_vs_k_children.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_query_time_vs_recall(self, results: List[Dict]):
        """Plot query time vs recall for all configurations."""
        fig, axes = plt.subplots(1, len(self.k_values), figsize=(5*len(self.k_values), 4))
        if len(self.k_values) == 1:
            axes = [axes]
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            # Filter results for this k
            filtered_results = [r for r in results if r['k'] == k]
            
            if filtered_results:
                recalls = [r['recall_at_k'] for r in filtered_results]
                query_times = [r['avg_query_time_ms'] for r in filtered_results]
                configs = [r['configuration'] for r in filtered_results]
                
                scatter = ax.scatter(recalls, query_times, alpha=0.7)
                
                # Add configuration labels for extreme points
                for j, (recall, time_ms, config) in enumerate(zip(recalls, query_times, configs)):
                    if recall > 0.8 or time_ms > np.percentile(query_times, 90):
                        ax.annotate(config, (recall, time_ms), fontsize=8, alpha=0.8)
            
            ax.set_xlabel(f'Recall@{k}')
            ax.set_ylabel('Query Time (ms)')
            ax.set_title(f'Query Time vs Recall@{k}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'query_time_vs_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_table(self, results: List[Dict]):
        """Generate a summary table of the best configurations."""
        # Find best configurations for each k
        best_configs = {}
        
        for k in self.k_values:
            k_results = [r for r in results if r['k'] == k]
            if k_results:
                # Sort by recall (descending), then by query time (ascending)
                best = max(k_results, key=lambda x: (x['recall_at_k'], -x['avg_query_time_ms']))
                best_configs[k] = best
        
        # Create summary
        summary = {
            'best_configurations': best_configs,
            'experiment_summary': {
                'total_configurations_tested': len(results),
                'dataset_size': self.dataset_size,
                'query_size': self.query_size,
                'dimension': self.dim,
                'parent_level': self.parent_level
            }
        }
        
        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self._convert_numpy_types(summary), f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        for k, config in best_configs.items():
            print(f"\nBest configuration for Recall@{k}:")
            print(f"  Configuration: {config['configuration']}")
            print(f"  Recall@{k}: {config['recall_at_k']:.4f}")
            print(f"  Query time: {config['avg_query_time_ms']:.2f} ms")
            print(f"  Total correct: {config['total_correct']}/{config['total_expected']}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description='HNSW Hybrid System Experiment')
    parser.add_argument('--dataset_size', type=int, default=10000,
                       help='Size of the dataset (default: 10000 for demo)')
    parser.add_argument('--query_size', type=int, default=100,
                       help='Size of the query set (default: 100 for demo)')
    parser.add_argument('--dim', type=int, default=128,
                       help='Vector dimension (default: 128)')
    parser.add_argument('--parent_level', type=int, default=2,
                       help='HNSW level to extract parent nodes from (default: 2)')
    parser.add_argument('--k_children', type=int, nargs='+', default=[500, 1000, 2000],
                       help='k_children values to test (default: [500, 1000, 2000])')
    parser.add_argument('--n_probe', type=int, nargs='+', default=[5, 10, 20, 50],
                       help='n_probe values to test (default: [5, 10, 20, 50])')
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 50, 100],
                       help='k values for recall evaluation (default: [10, 50, 100])')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                       help='Output directory for results (default: experiment_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = HNSWExperimentRunner(
        dataset_size=args.dataset_size,
        query_size=args.query_size,
        dim=args.dim,
        parent_level=args.parent_level,
        k_children_values=args.k_children,
        n_probe_values=args.n_probe,
        k_values=args.k_values,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    runner.run_full_experiment()


if __name__ == "__main__":
    main()
