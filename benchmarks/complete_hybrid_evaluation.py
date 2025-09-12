#!/usr/bin/env python3
"""
Complete HNSW Hybrid Two-Stage Retrieval System Evaluation
==========================================================

This script implements the complete project action guide for HNSW improvements
with comprehensive evaluation and parameter sweeps. Legacy standalone scripts
(`optimized_hybrid_hnsw.py`, `experiment_runner.py`, `parameter_tuning.py`,
`demo_hybrid_fix.py`) have been removed; their functionality is merged here
and in `hnsw_hybrid_evaluation.py`.

Project Phases:
1. Project Objectives and Core Concept Definition
2. Preparation and Baseline Construction  
3. Building the Custom Parent-Child Index Structure
4. Implementing the Two-Stage Search Logic
5. Experimental Evaluation

Author: HankyZhang
Date: September 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import json
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from hnsw_hybrid_evaluation import HybridHNSWIndex, RecallEvaluator, generate_synthetic_dataset, create_query_set


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""
    # Dataset configuration
    dataset_size: int = 600000  # 600K for comprehensive test (can scale to 6M)
    vector_dim: int = 128
    n_queries: int = 10000
    
    # Evaluation parameters
    k_values: List[int] = None  # [5, 10, 20, 50]
    k_children_values: List[int] = None  # [500, 1000, 1500, 2000]
    n_probe_values: List[int] = None  # [5, 10, 15, 20, 25]
    
    # HNSW parameters
    m: int = 16
    ef_construction: int = 200
    target_level: int = 2
    
    # Experimental settings
    random_seed: int = 42
    save_results: bool = True
    results_dir: str = "evaluation_results"
    
    def __post_init__(self):
        """Set default values for list parameters."""
        if self.k_values is None:
            self.k_values = [5, 10, 20, 50]
        if self.k_children_values is None:
            self.k_children_values = [500, 1000, 1500, 2000]
        if self.n_probe_values is None:
            self.n_probe_values = [5, 10, 15, 20, 25]


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator implementing the complete project action guide.
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize with evaluation configuration."""
        self.config = config
        self.dataset = None
        self.query_set = None
        self.ground_truth = {}
        self.results = []
        
        # Create results directory
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
    def phase1_objectives_and_concepts(self):
        """
        Phase 1: Project Objectives and Core Concept Definition
        """
        print("=" * 80)
        print("PHASE 1: PROJECT OBJECTIVES AND CORE CONCEPT DEFINITION")
        print("=" * 80)
        
        objectives = {
            "core_objective": "Validate recall performance of hybrid HNSW in plaintext environment",
            "approach": "Two-stage retrieval system",
            "stage_1": "Parent Layer - Extract high-level nodes as cluster centers",
            "stage_2": "Child Layer - Pre-computed neighbor sets for fine search",
            "focus": "Retrieval accuracy (ignoring encryption overhead)",
            "target_scale": f"{self.config.dataset_size:,} vectors"
        }
        
        print("Core Objectives:")
        for key, value in objectives.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\\nDataset Configuration:")
        print(f"  Dataset Size: {self.config.dataset_size:,} vectors")
        print(f"  Vector Dimension: {self.config.vector_dim}")
        print(f"  Query Set Size: {self.config.n_queries:,}")
        print(f"  Evaluation K values: {self.config.k_values}")
        
        # Save configuration
        if self.config.save_results:
            config_path = os.path.join(self.config.results_dir, "evaluation_config.json")
            with open(config_path, 'w') as f:
                # Convert config to dict for JSON serialization
                config_dict = {
                    "dataset_size": self.config.dataset_size,
                    "vector_dim": self.config.vector_dim,
                    "n_queries": self.config.n_queries,
                    "k_values": self.config.k_values,
                    "k_children_values": self.config.k_children_values,
                    "n_probe_values": self.config.n_probe_values,
                    "m": self.config.m,
                    "ef_construction": self.config.ef_construction,
                    "target_level": self.config.target_level,
                    "objectives": objectives
                }
                json.dump(config_dict, f, indent=2)
            print(f"Configuration saved to: {config_path}")
        
        return objectives
    
    def phase2_preparation_and_baseline(self):
        """
        Phase 2: Preparation and Baseline Construction
        """
        print("\\n" + "=" * 80)
        print("PHASE 2: PREPARATION AND BASELINE CONSTRUCTION")
        print("=" * 80)
        
        # Step 1: Data Preparation
        print("Step 1: Data Preparation")
        print(f"Generating dataset with {self.config.dataset_size:,} vectors...")
        start_time = time.time()
        
        self.dataset = generate_synthetic_dataset(
            n_vectors=self.config.dataset_size,
            dim=self.config.vector_dim
        )
        
        dataset_time = time.time() - start_time
        print(f"Dataset generated in {dataset_time:.2f} seconds")
        
        # Step 2: Query Set Creation
        print(f"\\nCreating query set with {self.config.n_queries:,} queries...")
        self.query_set = create_query_set(self.dataset, self.config.n_queries)
        
        # Step 3: Baseline Ground Truth Computation
        print("\\nStep 2: Baseline Ground Truth Computation")
        print("Computing ground truth using brute force search...")
        
        evaluator = RecallEvaluator(self.dataset)
        
        # Compute ground truth for all k values
        max_k = max(self.config.k_values)
        start_time = time.time()
        self.ground_truth = evaluator.compute_ground_truth(self.query_set, k=max_k)
        gt_time = time.time() - start_time
        
        print(f"Ground truth computed in {gt_time:.2f} seconds")
        
        # Save ground truth if configured
        if self.config.save_results:
            gt_path = os.path.join(self.config.results_dir, "ground_truth.pkl")
            with open(gt_path, 'wb') as f:
                pickle.dump(self.ground_truth, f)
            print(f"Ground truth saved to: {gt_path}")
        
        # Statistics
        print(f"\\nDataset Statistics:")
        print(f"  Total vectors: {len(self.dataset):,}")
        print(f"  Query vectors: {len(self.query_set):,}")
        print(f"  Vector dimension: {self.config.vector_dim}")
        print(f"  Memory usage (approx): {len(self.dataset) * self.config.vector_dim * 4 / 1024**2:.1f} MB")
        
        return {
            "dataset_size": len(self.dataset),
            "query_size": len(self.query_set),
            "dataset_time": dataset_time,
            "ground_truth_time": gt_time
        }
    
    def phase3_build_parent_child_structure(self, k_children: int, target_level: int = None) -> HybridHNSWIndex:
        """
        Phase 3: Building the Custom Parent-Child Index Structure
        """
        print(f"\\n" + "=" * 80)
        print("PHASE 3: BUILDING CUSTOM PARENT-CHILD INDEX STRUCTURE")
        print("=" * 80)
        
        if target_level is None:
            target_level = self.config.target_level
        
        print(f"Building hybrid index with k_children={k_children}, target_level={target_level}")
        
        # Step 1: Initialize hybrid index
        hybrid_index = HybridHNSWIndex(k_children=k_children)
        
        # Step 2: Build base HNSW index
        print("\\nStep 1: Building Base HNSW Index")
        start_time = time.time()
        hybrid_index.build_base_index(
            dataset=self.dataset,
            m=self.config.m,
            ef_construction=self.config.ef_construction
        )
        base_build_time = time.time() - start_time
        
        # Step 3: Extract parent nodes from target level
        print(f"\\nStep 2: Extracting Parent Nodes from Level {target_level}")
        start_time = time.time()
        parent_ids = hybrid_index.extract_parent_nodes(target_level=target_level)
        parent_extract_time = time.time() - start_time
        
        # Step 4: Build parent-child mapping
        print(f"\\nStep 3: Building Parent-Child Mapping")
        start_time = time.time()
        parent_child_map = hybrid_index.build_parent_child_mapping()
        mapping_time = time.time() - start_time
        
        # Statistics
        total_children = sum(len(children) for children in parent_child_map.values())
        avg_children = total_children / len(parent_child_map) if parent_child_map else 0
        
        structure_stats = {
            "base_build_time": base_build_time,
            "parent_extract_time": parent_extract_time,
            "mapping_time": mapping_time,
            "total_build_time": base_build_time + parent_extract_time + mapping_time,
            "num_parents": len(parent_ids),
            "total_children": total_children,
            "avg_children_per_parent": avg_children,
            "coverage_ratio": total_children / len(self.dataset),
            "target_level": target_level,
            "k_children": k_children
        }
        
        print(f"\\nStructure Statistics:")
        for key, value in structure_stats.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return hybrid_index, structure_stats
    
    def phase4_implement_two_stage_search(self, hybrid_index: HybridHNSWIndex, n_probe: int):
        """
        Phase 4: Implementing the Two-Stage Search Logic
        """
        print(f"\\n" + "=" * 80)
        print("PHASE 4: IMPLEMENTING TWO-STAGE SEARCH LOGIC")
        print("=" * 80)
        
        print(f"Configuring search with n_probe={n_probe}")
        hybrid_index.n_probe = n_probe
        
        # Test search functionality with a sample query
        print("\\nTesting search functionality...")
        sample_query_id = list(self.query_set.keys())[0]
        sample_query = self.query_set[sample_query_id]
        
        # Stage 1: Parent search
        print("Stage 1: Coarse Search (Parent Layer)")
        start_time = time.time()
        closest_parents = hybrid_index._find_closest_parents(sample_query)
        stage1_time = time.time() - start_time
        
        print(f"  Found {len(closest_parents)} closest parents in {stage1_time:.6f}s")
        print(f"  Parent IDs: {closest_parents[:5]}{'...' if len(closest_parents) > 5 else ''}")
        
        # Stage 2: Child search
        print("\\nStage 2: Fine Search (Child Layer)")
        start_time = time.time()
        
        # Collect candidates
        candidate_ids = set()
        for parent_id in closest_parents:
            if parent_id in hybrid_index.parent_child_map:
                candidate_ids.update(hybrid_index.parent_child_map[parent_id])
        candidate_ids.update(closest_parents)
        
        stage2_collection_time = time.time() - start_time
        
        # Perform final search in candidate set
        start_time = time.time()
        final_results = hybrid_index.search(sample_query, k=10)
        total_search_time = time.time() - start_time
        
        print(f"  Collected {len(candidate_ids)} candidates in {stage2_collection_time:.6f}s")
        print(f"  Total search time: {total_search_time:.6f}s")
        print(f"  Final results: {len(final_results)} neighbors found")
        
        search_stats = {
            "n_probe": n_probe,
            "stage1_time": stage1_time,
            "stage2_collection_time": stage2_collection_time,
            "total_search_time": total_search_time,
            "num_candidates": len(candidate_ids),
            "candidate_ratio": len(candidate_ids) / len(self.dataset)
        }
        
        print(f"\\nSearch Statistics:")
        for key, value in search_stats.items():
            if isinstance(value, float) and 'time' in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.6f}s")
            elif isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return search_stats
    
    def phase5_experimental_evaluation(self, hybrid_index: HybridHNSWIndex, k: int) -> Dict:
        """
        Phase 5: Experimental Evaluation
        """
        print(f"\\n" + "=" * 80)
        print("PHASE 5: EXPERIMENTAL EVALUATION")
        print("=" * 80)
        
        print(f"Evaluating recall@{k} performance...")
        
        # Use existing ground truth
        evaluator = RecallEvaluator(self.dataset)
        evaluator.ground_truth_cache = {
            qid: gt_results[:k] for qid, gt_results in self.ground_truth.items()
        }
        
        # Evaluate recall
        start_time = time.time()
        results = evaluator.evaluate_recall(hybrid_index, self.query_set, k=k)
        evaluation_time = time.time() - start_time
        
        # Add timing information
        results['evaluation_time'] = evaluation_time
        results['build_time'] = hybrid_index.build_time
        
        # Calculate additional metrics
        if hybrid_index.search_times:
            results['min_query_time'] = min(hybrid_index.search_times)
            results['max_query_time'] = max(hybrid_index.search_times)
            results['std_query_time'] = np.std(hybrid_index.search_times)
        
        print(f"\\nEvaluation Results for k={k}:")
        print(f"  Recall@{k}: {results['recall@k']:.4f}")
        print(f"  Average Query Time: {results['avg_query_time']:.6f}s")
        print(f"  Total Evaluation Time: {evaluation_time:.2f}s")
        print(f"  Parameters: k_children={results['k_children']}, n_probe={results['n_probe']}")
        
        return results
    
    def run_parameter_sweep(self):
        """
        Run comprehensive parameter sweep across k_children and n_probe values.
        """
        print(f"\\n" + "=" * 80)
        print("COMPREHENSIVE PARAMETER SWEEP")
        print("=" * 80)
        
        # Ensure dataset is available
        if self.dataset is None or self.query_set is None:
            print("Dataset not prepared. Running Phase 2 preparation...")
            self.phase2_preparation_and_baseline()
        
        print(f"Testing combinations:")
        print(f"  k_children: {self.config.k_children_values}")
        print(f"  n_probe: {self.config.n_probe_values}")
        print(f"  k values: {self.config.k_values}")
        
        all_results = []
        total_combinations = len(self.config.k_children_values) * len(self.config.n_probe_values) * len(self.config.k_values)
        combination_count = 0
        
        for k_children in self.config.k_children_values:
            print(f"\\n{'='*60}")
            print(f"TESTING k_children = {k_children}")
            print(f"{'='*60}")
            
            # Build index structure once per k_children
            hybrid_index, structure_stats = self.phase3_build_parent_child_structure(k_children)
            
            for n_probe in self.config.n_probe_values:
                print(f"\\n{'-'*40}")
                print(f"Testing n_probe = {n_probe}")
                print(f"{'-'*40}")
                
                # Configure search parameters
                search_stats = self.phase4_implement_two_stage_search(hybrid_index, n_probe)
                
                for k in self.config.k_values:
                    combination_count += 1
                    print(f"\\nEvaluating k={k} ({combination_count}/{total_combinations})...")
                    
                    # Evaluate performance
                    eval_results = self.phase5_experimental_evaluation(hybrid_index, k)
                    
                    # Combine all statistics
                    combined_results = {
                        **structure_stats,
                        **search_stats,
                        **eval_results,
                        'combination_id': combination_count
                    }
                    
                    all_results.append(combined_results)
                    
                    # Print summary
                    print(f"  Result: Recall@{k}={eval_results['recall@k']:.4f}, "
                          f"Time={eval_results['avg_query_time']:.6f}s")
        
        # Save all results
        if self.config.save_results:
            results_path = os.path.join(self.config.results_dir, "parameter_sweep_results.json")
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\\nComplete results saved to: {results_path}")
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """
        Analyze and summarize the parameter sweep results.
        """
        print(f"\\n" + "=" * 80)
        print("RESULTS ANALYSIS")
        print("=" * 80)
        
        if not self.results:
            print("No results to analyze. Run parameter sweep first.")
            return
        
        # Group results by k value
        k_groups = {}
        for result in self.results:
            k = result['k']
            if k not in k_groups:
                k_groups[k] = []
            k_groups[k].append(result)
        
        analysis = {}
        
        for k in sorted(k_groups.keys()):
            print(f"\\n{'='*40}")
            print(f"ANALYSIS FOR k={k}")
            print(f"{'='*40}")
            
            results_k = k_groups[k]
            
            # Find best configuration by recall
            best_recall = max(results_k, key=lambda x: x['recall@k'])
            best_time = min(results_k, key=lambda x: x['avg_query_time'])
            
            # Find balanced configuration (recall >= 0.8 of best, minimize time)
            best_recall_value = best_recall['recall@k']
            threshold = 0.8 * best_recall_value
            good_recall_results = [r for r in results_k if r['recall@k'] >= threshold]
            
            if good_recall_results:
                balanced = min(good_recall_results, key=lambda x: x['avg_query_time'])
            else:
                balanced = best_recall
            
            print(f"Best Recall Configuration:")
            print(f"  k_children={best_recall['k_children']}, n_probe={best_recall['n_probe']}")
            print(f"  Recall@{k}: {best_recall['recall@k']:.4f}")
            print(f"  Query Time: {best_recall['avg_query_time']:.6f}s")
            
            print(f"\\nFastest Configuration:")
            print(f"  k_children={best_time['k_children']}, n_probe={best_time['n_probe']}")
            print(f"  Recall@{k}: {best_time['recall@k']:.4f}")
            print(f"  Query Time: {best_time['avg_query_time']:.6f}s")
            
            print(f"\\nBalanced Configuration (≥{threshold:.3f} recall):")
            print(f"  k_children={balanced['k_children']}, n_probe={balanced['n_probe']}")
            print(f"  Recall@{k}: {balanced['recall@k']:.4f}")
            print(f"  Query Time: {balanced['avg_query_time']:.6f}s")
            
            analysis[k] = {
                'best_recall': best_recall,
                'fastest': best_time,
                'balanced': balanced,
                'num_configurations': len(results_k)
            }
        
        # Overall statistics
        print(f"\\n{'='*40}")
        print("OVERALL STATISTICS")
        print(f"{'='*40}")
        
        all_recalls = [r['recall@k'] for r in self.results]
        all_times = [r['avg_query_time'] for r in self.results]
        
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Recall range: {min(all_recalls):.4f} - {max(all_recalls):.4f}")
        print(f"Query time range: {min(all_times):.6f}s - {max(all_times):.6f}s")
        print(f"Average recall: {np.mean(all_recalls):.4f}")
        print(f"Average query time: {np.mean(all_times):.6f}s")
        
        # Save analysis
        if self.config.save_results:
            analysis_path = os.path.join(self.config.results_dir, "results_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\\nAnalysis saved to: {analysis_path}")
        
        return analysis
    
    def run_complete_evaluation(self):
        """
        Run the complete evaluation pipeline following the project action guide.
        """
        print("HNSW HYBRID TWO-STAGE RETRIEVAL SYSTEM")
        print("Complete Evaluation Pipeline")
        print("=" * 80)
        
        start_time = time.time()
        
        # Execute all phases
        objectives = self.phase1_objectives_and_concepts()
        preparation_stats = self.phase2_preparation_and_baseline()
        sweep_results = self.run_parameter_sweep()
        analysis = self.analyze_results()
        
        total_time = time.time() - start_time
        
        # Final summary
        print(f"\\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        print(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Configurations tested: {len(sweep_results)}")
        print(f"Results directory: {self.config.results_dir}")
        
        summary = {
            'objectives': objectives,
            'preparation_stats': preparation_stats,
            'total_time': total_time,
            'num_configurations': len(sweep_results),
            'results_directory': self.config.results_dir
        }
        
        # Save final summary
        if self.config.save_results:
            summary_path = os.path.join(self.config.results_dir, "evaluation_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Evaluation summary saved to: {summary_path}")
        
        return summary


def main():
    """
    Main function to run the complete evaluation.
    """
    # Configuration for comprehensive evaluation
    config = EvaluationConfig(
        dataset_size=100000,  # Start with 100K for faster testing
        vector_dim=128,
        n_queries=2000,
        k_values=[5, 10, 20],
        k_children_values=[500, 1000, 1500],
        n_probe_values=[5, 10, 15, 20],
        save_results=True,
        results_dir="hybrid_hnsw_results"
    )
    
    # Run complete evaluation
    evaluator = ComprehensiveEvaluator(config)
    summary = evaluator.run_complete_evaluation()
    
    print("\\nEvaluation completed successfully!")
    return summary


if __name__ == "__main__":
    summary = main()
