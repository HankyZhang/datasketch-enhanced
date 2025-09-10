#!/usr/bin/env python3
"""
HNSW Hybrid System - Parameter Tuning and Analysis

This script focuses specifically on parameter tuning for the HNSW hybrid system,
implementing systematic parameter sweeps and analysis to find optimal configurations.

Usage:
    python parameter_tuning.py --dataset_size 100000 --query_size 1000 --dim 128
"""

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
from itertools import product

from datasketch.hnsw import HNSW
from hnsw_hybrid import HNSWHybrid, HNSWEvaluator, create_synthetic_dataset, create_query_set


class ParameterTuner:
    """
    Parameter tuning system for HNSW hybrid index.
    """
    
    def __init__(
        self,
        dataset_size: int = 100000,
        query_size: int = 1000,
        dim: int = 128,
        parent_level: int = 2,
        output_dir: str = "parameter_tuning_results",
        seed: int = 42
    ):
        """
        Initialize the parameter tuner.
        
        Args:
            dataset_size: Size of the dataset
            query_size: Size of the query set
            dim: Vector dimension
            parent_level: HNSW level to extract parent nodes from
            output_dir: Directory to save results
            seed: Random seed
        """
        self.dataset_size = dataset_size
        self.query_size = query_size
        self.dim = dim
        self.parent_level = parent_level
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Data structures
        self.dataset: Optional[np.ndarray] = None
        self.query_vectors: Optional[np.ndarray] = None
        self.query_ids: Optional[List[int]] = None
        self.base_index: Optional[HNSW] = None
        self.evaluator: Optional[HNSWEvaluator] = None
        
        # Results storage
        self.tuning_results: List[Dict] = []
        self.best_configs: Dict = {}
    
    def prepare_data(self):
        """Prepare dataset and build base index."""
        print("Preparing data and building base index...")
        
        # Create synthetic dataset
        self.dataset = create_synthetic_dataset(self.dataset_size, self.dim, self.seed)
        
        # Create query set
        self.query_vectors, self.query_ids = create_query_set(
            self.dataset, self.query_size, self.seed + 1
        )
        
        # Build base HNSW index
        distance_func = lambda x, y: np.linalg.norm(x - y)
        self.base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)
        
        # Insert vectors (excluding queries)
        for i, vector in enumerate(self.dataset):
            if i not in self.query_ids:
                self.base_index.insert(i, vector)
        
        # Initialize evaluator
        self.evaluator = HNSWEvaluator(self.dataset, self.query_vectors, self.query_ids)
        
        print(f"Data prepared: {len(self.dataset)} vectors, {len(self.query_vectors)} queries")
        print(f"Base index built: {len(self.base_index)} vectors, {len(self.base_index._graphs)} layers")
    
    def tune_parameters(
        self,
        k_children_range: Tuple[int, int, int] = (100, 2000, 100),
        n_probe_range: Tuple[int, int, int] = (1, 50, 1),
        k_values: List[int] = [10, 50, 100],
        max_configurations: int = 100
    ):
        """
        Perform systematic parameter tuning.
        
        Args:
            k_children_range: (start, stop, step) for k_children values
            n_probe_range: (start, stop, step) for n_probe values
            k_values: List of k values for recall evaluation
            max_configurations: Maximum number of configurations to test
        """
        print("\n" + "=" * 80)
        print("PARAMETER TUNING")
        print("=" * 80)
        
        # Generate parameter combinations
        k_children_values = list(range(*k_children_range))
        n_probe_values = list(range(*n_probe_range))
        
        # Limit number of configurations if too many
        total_configs = len(k_children_values) * len(n_probe_values) * len(k_values)
        if total_configs > max_configurations:
            print(f"Too many configurations ({total_configs}). Sampling {max_configurations}...")
            # Sample configurations
            k_children_values = np.linspace(k_children_range[0], k_children_range[1]-1, 
                                          min(len(k_children_values), max_configurations//len(n_probe_values)//len(k_values))).astype(int)
            n_probe_values = np.linspace(n_probe_range[0], n_probe_range[1]-1,
                                       min(len(n_probe_values), max_configurations//len(k_children_values)//len(k_values))).astype(int)
        
        print(f"Testing {len(k_children_values)} k_children values: {k_children_values}")
        print(f"Testing {len(n_probe_values)} n_probe values: {n_probe_values}")
        print(f"Testing {len(k_values)} k values: {k_values}")
        
        # Compute ground truth for all k values
        ground_truths = {}
        for k in k_values:
            print(f"\nComputing ground truth for k={k}...")
            start_time = time.time()
            ground_truths[k] = self.evaluator.compute_ground_truth(k, self.base_index._distance_func)
            print(f"Ground truth computed in {time.time() - start_time:.2f}s")
        
        # Test all parameter combinations
        config_count = 0
        total_configs = len(k_children_values) * len(n_probe_values) * len(k_values)
        
        for k_children in k_children_values:
            print(f"\nBuilding hybrid index with k_children={k_children}")
            start_time = time.time()
            
            # Build hybrid index
            hybrid_index = HNSWHybrid(
                base_index=self.base_index,
                parent_level=self.parent_level,
                k_children=k_children,
                distance_func=self.base_index._distance_func
            )
            
            build_time = time.time() - start_time
            print(f"Hybrid index built in {build_time:.2f}s")
            
            # Test all n_probe and k combinations
            for n_probe in n_probe_values:
                for k in k_values:
                    config_count += 1
                    print(f"  [{config_count}/{total_configs}] Testing k_children={k_children}, "
                          f"n_probe={n_probe}, k={k}")
                    
                    # Evaluate configuration
                    result = self.evaluator.evaluate_recall(
                        hybrid_index, k=k, n_probe=n_probe, ground_truth=ground_truths[k]
                    )
                    
                    # Add configuration info
                    result.update({
                        'k_children': k_children,
                        'build_time': build_time,
                        'configuration_id': f"k_children={k_children}_n_probe={n_probe}_k={k}"
                    })
                    
                    self.tuning_results.append(result)
                    
                    print(f"    Recall@{k}: {result['recall_at_k']:.4f}, "
                          f"Query time: {result['avg_query_time_ms']:.2f}ms")
        
        # Find best configurations
        self._find_best_configurations(k_values)
        
        # Save results
        self._save_tuning_results()
        
        # Generate analysis
        self._generate_tuning_analysis()
    
    def _find_best_configurations(self, k_values: List[int]):
        """Find the best configurations for each k value."""
        print("\n" + "-" * 60)
        print("FINDING BEST CONFIGURATIONS")
        print("-" * 60)
        
        for k in k_values:
            # Filter results for this k
            k_results = [r for r in self.tuning_results if r['k'] == k]
            
            if k_results:
                # Sort by recall (descending), then by query time (ascending)
                best = max(k_results, key=lambda x: (x['recall_at_k'], -x['avg_query_time_ms']))
                self.best_configs[k] = best
                
                print(f"\nBest configuration for Recall@{k}:")
                print(f"  k_children: {best['k_children']}")
                print(f"  n_probe: {best['n_probe']}")
                print(f"  Recall@{k}: {best['recall_at_k']:.4f}")
                print(f"  Query time: {best['avg_query_time_ms']:.2f} ms")
                print(f"  Total correct: {best['total_correct']}/{best['total_expected']}")
    
    def _save_tuning_results(self):
        """Save tuning results to files."""
        print("\n" + "-" * 60)
        print("SAVING TUNING RESULTS")
        print("-" * 60)
        
        # Save all results
        results_file = self.output_dir / "tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump(self._convert_numpy_types(self.tuning_results), f, indent=2)
        
        # Save best configurations
        best_file = self.output_dir / "best_configurations.json"
        with open(best_file, 'w') as f:
            json.dump(self._convert_numpy_types(self.best_configs), f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(self.tuning_results)
        csv_file = self.output_dir / "tuning_results.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {results_file}")
        print(f"Best configurations saved to {best_file}")
        print(f"CSV results saved to {csv_file}")
    
    def _generate_tuning_analysis(self):
        """Generate comprehensive analysis of tuning results."""
        print("\n" + "-" * 60)
        print("GENERATING TUNING ANALYSIS")
        print("-" * 60)
        
        # Create analysis plots
        self._plot_parameter_heatmaps()
        self._plot_recall_landscapes()
        self._plot_efficiency_curves()
        self._generate_parameter_importance_analysis()
        
        print("Analysis plots generated")
    
    def _plot_parameter_heatmaps(self):
        """Create heatmaps showing recall vs parameter combinations."""
        df = pd.DataFrame(self.tuning_results)
        
        for k in df['k'].unique():
            k_df = df[df['k'] == k]
            
            # Create pivot table for heatmap
            pivot = k_df.pivot_table(
                values='recall_at_k', 
                index='k_children', 
                columns='n_probe', 
                aggfunc='mean'
            )
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': f'Recall@{k}'})
            plt.title(f'Recall@{k} Heatmap: k_children vs n_probe')
            plt.xlabel('n_probe')
            plt.ylabel('k_children')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'recall_heatmap_k{k}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_recall_landscapes(self):
        """Create 3D surface plots of recall landscapes."""
        df = pd.DataFrame(self.tuning_results)
        
        for k in df['k'].unique():
            k_df = df[df['k'] == k]
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create surface
            k_children = k_df['k_children'].values
            n_probe = k_df['n_probe'].values
            recall = k_df['recall_at_k'].values
            
            # Create meshgrid for surface
            k_children_unique = sorted(k_df['k_children'].unique())
            n_probe_unique = sorted(k_df['n_probe'].unique())
            K_children, N_probe = np.meshgrid(k_children_unique, n_probe_unique)
            
            # Interpolate recall values
            Recall = np.zeros_like(K_children)
            for i, kc in enumerate(k_children_unique):
                for j, np_val in enumerate(n_probe_unique):
                    mask = (k_df['k_children'] == kc) & (k_df['n_probe'] == np_val)
                    if mask.any():
                        Recall[j, i] = k_df[mask]['recall_at_k'].iloc[0]
            
            # Plot surface
            surf = ax.plot_surface(K_children, N_probe, Recall, cmap='viridis', alpha=0.8)
            ax.set_xlabel('k_children')
            ax.set_ylabel('n_probe')
            ax.set_zlabel(f'Recall@{k}')
            ax.set_title(f'Recall@{k} Landscape')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'recall_landscape_k{k}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_efficiency_curves(self):
        """Plot efficiency curves (recall vs query time)."""
        df = pd.DataFrame(self.tuning_results)
        
        fig, axes = plt.subplots(1, len(df['k'].unique()), figsize=(5*len(df['k'].unique()), 4))
        if len(df['k'].unique()) == 1:
            axes = [axes]
        
        for i, k in enumerate(df['k'].unique()):
            ax = axes[i]
            k_df = df[df['k'] == k]
            
            # Color by k_children
            scatter = ax.scatter(k_df['avg_query_time_ms'], k_df['recall_at_k'], 
                               c=k_df['k_children'], cmap='viridis', alpha=0.7)
            
            ax.set_xlabel('Query Time (ms)')
            ax.set_ylabel(f'Recall@{k}')
            ax.set_title(f'Efficiency Curve for k={k}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('k_children')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_parameter_importance_analysis(self):
        """Analyze parameter importance using correlation analysis."""
        df = pd.DataFrame(self.tuning_results)
        
        # Calculate correlations
        correlations = {}
        for k in df['k'].unique():
            k_df = df[df['k'] == k]
            
            corr_k_children = k_df['k_children'].corr(k_df['recall_at_k'])
            corr_n_probe = k_df['n_probe'].corr(k_df['recall_at_k'])
            corr_query_time = k_df['avg_query_time_ms'].corr(k_df['recall_at_k'])
            
            correlations[k] = {
                'k_children_vs_recall': corr_k_children,
                'n_probe_vs_recall': corr_n_probe,
                'query_time_vs_recall': corr_query_time
            }
        
        # Create correlation plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_values = list(correlations.keys())
        k_children_corr = [correlations[k]['k_children_vs_recall'] for k in k_values]
        n_probe_corr = [correlations[k]['n_probe_vs_recall'] for k in k_values]
        query_time_corr = [correlations[k]['query_time_vs_recall'] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.25
        
        ax.bar(x - width, k_children_corr, width, label='k_children vs recall', alpha=0.8)
        ax.bar(x, n_probe_corr, width, label='n_probe vs recall', alpha=0.8)
        ax.bar(x + width, query_time_corr, width, label='query_time vs recall', alpha=0.8)
        
        ax.set_xlabel('k values')
        ax.set_ylabel('Correlation with Recall')
        ax.set_title('Parameter Importance Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation analysis
        corr_file = self.output_dir / "parameter_correlations.json"
        with open(corr_file, 'w') as f:
            json.dump(correlations, f, indent=2)
        
        print("Parameter importance analysis completed")
    
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
    """Main function to run parameter tuning."""
    parser = argparse.ArgumentParser(description='HNSW Hybrid System Parameter Tuning')
    parser.add_argument('--dataset_size', type=int, default=100000,
                       help='Size of the dataset (default: 100000)')
    parser.add_argument('--query_size', type=int, default=1000,
                       help='Size of the query set (default: 1000)')
    parser.add_argument('--dim', type=int, default=128,
                       help='Vector dimension (default: 128)')
    parser.add_argument('--parent_level', type=int, default=2,
                       help='HNSW level to extract parent nodes from (default: 2)')
    parser.add_argument('--k_children_range', type=int, nargs=3, default=[100, 2000, 100],
                       help='k_children range: start stop step (default: 100 2000 100)')
    parser.add_argument('--n_probe_range', type=int, nargs=3, default=[1, 50, 1],
                       help='n_probe range: start stop step (default: 1 50 1)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 50, 100],
                       help='k values for recall evaluation (default: [10, 50, 100])')
    parser.add_argument('--max_configurations', type=int, default=100,
                       help='Maximum number of configurations to test (default: 100)')
    parser.add_argument('--output_dir', type=str, default='parameter_tuning_results',
                       help='Output directory for results (default: parameter_tuning_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create and run parameter tuning
    tuner = ParameterTuner(
        dataset_size=args.dataset_size,
        query_size=args.query_size,
        dim=args.dim,
        parent_level=args.parent_level,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Prepare data
    tuner.prepare_data()
    
    # Run parameter tuning
    tuner.tune_parameters(
        k_children_range=tuple(args.k_children_range),
        n_probe_range=tuple(args.n_probe_range),
        k_values=args.k_values,
        max_configurations=args.max_configurations
    )
    
    print("\n" + "=" * 80)
    print("PARAMETER TUNING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    # Import seaborn for better plots
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Warning: seaborn not available. Install with 'pip install seaborn' for better plots.")
        sns = None
    
    main()
