#!/usr/bin/env python3
"""
1M SIFT Dataset Parameter Testing with Level 2 Constraints

Constraints:
- parent_level = 2 (fixed)
- n_probe <= 0.5 * number_of_parents
- Test full 1M base vectors + 10K queries
- Find optimal parameters for different recall targets
"""

import numpy as np
import struct
import time
import json
import os
from typing import Dict, List, Tuple, Optional
import gc

def read_fvecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .fvecs format files efficiently with progress tracking."""
    vectors = []
    count = 0
    
    print(f"Reading {filename}...")
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
            
            if count % 100000 == 0:
                print(f"  Progress: {count:,} vectors loaded...")
    
    print(f"  Completed: {count:,} vectors loaded")
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename: str, max_count: Optional[int] = None) -> np.ndarray:
    """Read .ivecs format files efficiently."""
    vectors = []
    count = 0
    
    print(f"Reading {filename}...")
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
    
    print(f"  Completed: {count:,} vectors loaded")
    return np.array(vectors, dtype=np.int32)

class SIFT1MParameterTester:
    """Parameter testing for 1M SIFT dataset with level 2 constraints."""
    
    def __init__(self, test_queries: int = 1000):
        self.test_queries = test_queries
        self.base_vectors = None
        self.query_vectors = None
        self.ground_truth = None
        
    def load_dataset(self):
        """Load the full 1M SIFT dataset."""
        print("🔄 Loading 1M SIFT Dataset")
        print("=" * 50)
        
        # Load full base vectors (1M)
        self.base_vectors = read_fvecs("sift/sift_base.fvecs")
        print(f"✅ Base vectors: {self.base_vectors.shape}")
        
        # Load query vectors
        self.query_vectors = read_fvecs("sift/sift_query.fvecs", self.test_queries)
        print(f"✅ Query vectors: {self.query_vectors.shape}")
        
        # Load ground truth
        self.ground_truth = read_ivecs("sift/sift_groundtruth.ivecs", self.test_queries)
        print(f"✅ Ground truth: {self.ground_truth.shape}")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Base vectors: {len(self.base_vectors):,}")
        print(f"   Query vectors: {len(self.query_vectors):,}")
        print(f"   Dimensions: {self.base_vectors.shape[1]}")
        print(f"   Total memory: ~{self.estimate_memory():.1f} GB")
        
    def estimate_memory(self):
        """Estimate memory usage in GB."""
        base_size = self.base_vectors.nbytes if self.base_vectors is not None else 0
        query_size = self.query_vectors.nbytes if self.query_vectors is not None else 0
        gt_size = self.ground_truth.nbytes if self.ground_truth is not None else 0
        return (base_size + query_size + gt_size) / (1024**3)
    
    def test_configuration(self, m: int, ef_construction: int, k_children: int) -> Optional[Dict]:
        """Test a specific parameter configuration."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing Configuration:")
        print(f"   m = {m}")
        print(f"   ef_construction = {ef_construction}")
        print(f"   k_children = {k_children}")
        print(f"   parent_level = 2 (fixed)")
        print(f"{'='*60}")
        
        try:
            from hnsw import HNSW
            from hybrid_hnsw import HNSWHybrid
            
            # Build base HNSW index
            print("🔨 Building base HNSW index...")
            distance_func = lambda x, y: np.linalg.norm(x - y)
            
            start_time = time.time()
            base_index = HNSW(distance_func=distance_func, m=m, ef_construction=ef_construction)
            
            # Build in batches to manage memory and show progress
            batch_size = 50000
            dataset = {}
            
            for i in range(0, len(self.base_vectors), batch_size):
                end_i = min(i + batch_size, len(self.base_vectors))
                print(f"  Building batch {i//batch_size + 1}: {i:,} to {end_i:,}")
                
                batch_dataset = {j: self.base_vectors[j] for j in range(i, end_i)}
                dataset.update(batch_dataset)
                base_index.update(batch_dataset)
                
                # Force garbage collection to manage memory
                if i > 0:
                    gc.collect()
            
            base_build_time = time.time() - start_time
            print(f"✅ Base HNSW built in {base_build_time:.2f}s ({base_build_time/60:.1f} min)")
            
            # Build hybrid structure
            print("🚀 Building hybrid structure...")
            start_time = time.time()
            
            hybrid_index = HNSWHybrid(
                base_index=base_index,
                parent_level=2,
                k_children=k_children,
                parent_child_method='exact'
            )
            
            hybrid_build_time = time.time() - start_time
            total_build_time = base_build_time + hybrid_build_time
            
            # Get structure statistics
            stats = hybrid_index.get_stats()
            num_parents = stats.get('num_parents', 0)
            num_children = stats.get('num_children', 0)
            avg_children = stats.get('avg_children_per_parent', 0)
            
            print(f"✅ Hybrid structure built in {hybrid_build_time:.2f}s")
            print(f"\n📊 Structure Statistics:")
            print(f"   Parents at level 2: {num_parents:,}")
            print(f"   Children mapped: {num_children:,}")
            print(f"   Avg children/parent: {avg_children:.1f}")
            
            # Calculate n_probe constraint
            max_n_probe = max(1, int(0.5 * num_parents))
            print(f"🎯 n_probe constraint: n_probe ≤ {max_n_probe} (0.5 × {num_parents})")
            
            # Determine n_probe values to test
            n_probe_candidates = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
            n_probe_values = [n for n in n_probe_candidates if n <= max_n_probe]
            
            if not n_probe_values:
                n_probe_values = [1]
            
            print(f"🧪 Testing n_probe values: {n_probe_values}")
            
            # Test different n_probe values
            results = {}
            
            for n_probe in n_probe_values:
                print(f"\n📈 Testing n_probe = {n_probe}...")
                
                query_times = []
                recalls_at_k = {1: [], 10: [], 100: []}
                
                # Test on subset of queries for each n_probe
                test_sample = min(100, len(self.query_vectors))
                
                for i in range(test_sample):
                    if i % 20 == 0:
                        print(f"  Query progress: {i}/{test_sample}")
                    
                    query = self.query_vectors[i]
                    
                    # Time the search
                    start_time = time.time()
                    search_results = hybrid_index.search(query, k=100, n_probe=n_probe)
                    query_time = time.time() - start_time
                    query_times.append(query_time)
                    
                    # Calculate recall
                    result_ids = [rid for rid, _ in search_results]
                    gt_ids = self.ground_truth[i]
                    
                    for k in [1, 10, 100]:
                        k_min = min(k, len(result_ids), len(gt_ids))
                        if k_min > 0:
                            recall_k = len(set(result_ids[:k_min]) & set(gt_ids[:k_min])) / k_min
                            recalls_at_k[k].append(recall_k)
                
                # Calculate averages
                avg_query_time = np.mean(query_times) * 1000  # Convert to ms
                avg_recalls = {k: np.mean(recalls) for k, recalls in recalls_at_k.items()}
                
                results[n_probe] = {
                    'query_time_ms': avg_query_time,
                    'recalls': avg_recalls,
                    'sample_size': test_sample
                }
                
                print(f"  ✅ Results: {avg_query_time:.2f}ms, "
                      f"R@1={avg_recalls[1]:.3f}, "
                      f"R@10={avg_recalls[10]:.3f}, "
                      f"R@100={avg_recalls[100]:.3f}")
            
            return {
                'config': {
                    'm': m,
                    'ef_construction': ef_construction,
                    'k_children': k_children,
                    'parent_level': 2
                },
                'build_times': {
                    'base_seconds': base_build_time,
                    'hybrid_seconds': hybrid_build_time,
                    'total_seconds': total_build_time,
                    'total_minutes': total_build_time / 60
                },
                'structure_stats': {
                    'num_parents': num_parents,
                    'num_children': num_children,
                    'avg_children_per_parent': avg_children,
                    'max_n_probe': max_n_probe
                },
                'performance_results': results,
                'dataset_info': {
                    'base_vectors': len(self.base_vectors),
                    'test_queries': test_sample,
                    'dimensions': self.base_vectors.shape[1]
                }
            }
            
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up memory
            if 'base_index' in locals():
                del base_index
            if 'hybrid_index' in locals():
                del hybrid_index
            if 'dataset' in locals():
                del dataset
            gc.collect()
    
    def run_parameter_sweep(self):
        """Run comprehensive parameter sweep for 1M dataset."""
        print("🎯 1M SIFT Parameter Sweep - Level 2 Optimization")
        print("=" * 70)
        
        # Parameter configurations optimized for level 2
        # Focus on configurations that provide good parent distribution at level 2
        configs = [
            # (m, ef_construction, k_children) - comments explain rationale
            (16, 200, 1000),   # Baseline: standard params
            (16, 200, 2000),   # More children per parent
            (16, 400, 1000),   # Higher quality base index
            (16, 400, 2000),   # High quality + more children
            (24, 200, 1500),   # Higher connectivity
            (24, 400, 1500),   # High connectivity + quality
            (32, 200, 1000),   # Very high connectivity
            (16, 600, 2000),   # Very high quality base
            (20, 300, 1500),   # Balanced approach
            (16, 800, 1000),   # Maximum quality base
        ]
        
        print(f"🧪 Testing {len(configs)} configurations...")
        print(f"📊 Each config will be tested with constraint: n_probe ≤ 0.5 × num_parents")
        
        all_results = []
        successful_configs = 0
        
        for i, (m, ef_construction, k_children) in enumerate(configs, 1):
            print(f"\n{'🔬' * 3} Configuration {i}/{len(configs)} {'🔬' * 3}")
            
            try:
                result = self.test_configuration(m, ef_construction, k_children)
                
                if result:
                    all_results.append(result)
                    successful_configs += 1
                    
                    # Show summary for this config
                    best_n_probe = max(result['performance_results'].keys(),
                                     key=lambda n: result['performance_results'][n]['recalls'][10])
                    best_perf = result['performance_results'][best_n_probe]
                    
                    print(f"\n✅ Configuration {i} Summary:")
                    print(f"   Best n_probe: {best_n_probe} (max: {result['structure_stats']['max_n_probe']})")
                    print(f"   Best Recall@10: {best_perf['recalls'][10]:.3f}")
                    print(f"   Query time: {best_perf['query_time_ms']:.2f}ms")
                    print(f"   Build time: {result['build_times']['total_minutes']:.1f} min")
                    print(f"   Parents: {result['structure_stats']['num_parents']:,}")
                
            except KeyboardInterrupt:
                print("\n⚠️  Testing interrupted by user")
                break
            except Exception as e:
                print(f"❌ Configuration {i} failed: {e}")
                continue
        
        # Analysis and recommendations
        if all_results:
            self.analyze_results(all_results, successful_configs, len(configs))
            self.save_results(all_results)
        else:
            print("❌ No successful configurations found!")
    
    def analyze_results(self, results: List[Dict], successful: int, total: int):
        """Analyze and present results."""
        print(f"\n{'='*70}")
        print(f"📊 COMPREHENSIVE ANALYSIS - 1M SIFT Dataset")
        print(f"{'='*70}")
        print(f"✅ Successful configurations: {successful}/{total}")
        
        # Extract best performance for each config
        config_summaries = []
        
        for result in results:
            config = result['config']
            
            # Find best n_probe for this config
            best_n_probe = max(result['performance_results'].keys(),
                             key=lambda n: result['performance_results'][n]['recalls'][10])
            best_perf = result['performance_results'][best_n_probe]
            
            config_summaries.append({
                'config_str': f"m={config['m']}, ef_c={config['ef_construction']}, k_ch={config['k_children']}",
                'config': config,
                'best_n_probe': best_n_probe,
                'max_n_probe': result['structure_stats']['max_n_probe'],
                'num_parents': result['structure_stats']['num_parents'],
                'recall_1': best_perf['recalls'][1],
                'recall_10': best_perf['recalls'][10],
                'recall_100': best_perf['recalls'][100],
                'query_time_ms': best_perf['query_time_ms'],
                'build_time_min': result['build_times']['total_minutes']
            })
        
        # Sort by different criteria
        print(f"\n🏆 TOP CONFIGURATIONS BY RECALL@10:")
        print(f"{'Rank':<4} {'Recall@10':<9} {'Query(ms)':<10} {'n_probe':<8} {'Parents':<8} {'Config':<30}")
        print("-" * 80)
        
        by_recall = sorted(config_summaries, key=lambda x: x['recall_10'], reverse=True)
        for i, summary in enumerate(by_recall[:5], 1):
            print(f"{i:<4} {summary['recall_10']:<9.3f} {summary['query_time_ms']:<10.2f} "
                  f"{summary['best_n_probe']:<8} {summary['num_parents']:<8} {summary['config_str']:<30}")
        
        print(f"\n⚡ TOP CONFIGURATIONS BY SPEED:")
        print(f"{'Rank':<4} {'Query(ms)':<10} {'Recall@10':<9} {'n_probe':<8} {'Parents':<8} {'Config':<30}")
        print("-" * 80)
        
        by_speed = sorted(config_summaries, key=lambda x: x['query_time_ms'])
        for i, summary in enumerate(by_speed[:5], 1):
            print(f"{i:<4} {summary['query_time_ms']:<10.2f} {summary['recall_10']:<9.3f} "
                  f"{summary['best_n_probe']:<8} {summary['num_parents']:<8} {summary['config_str']:<30}")
        
        # Recommendations by use case
        print(f"\n🎯 RECOMMENDATIONS BY USE CASE:")
        print("-" * 50)
        
        # Best overall (highest recall@10)
        best_overall = by_recall[0]
        print(f"🥇 Best Overall Performance:")
        print(f"   Configuration: {best_overall['config_str']}")
        print(f"   Optimal n_probe: {best_overall['best_n_probe']} (max: {best_overall['max_n_probe']})")
        print(f"   Performance: {best_overall['recall_10']:.1%} recall@10, {best_overall['query_time_ms']:.2f}ms")
        print(f"   Build time: {best_overall['build_time_min']:.1f} minutes")
        
        # Best for speed
        best_speed = by_speed[0]
        if best_speed != best_overall:
            print(f"\n🏃 Best for Speed:")
            print(f"   Configuration: {best_speed['config_str']}")
            print(f"   Optimal n_probe: {best_speed['best_n_probe']}")
            print(f"   Performance: {best_speed['query_time_ms']:.2f}ms, {best_speed['recall_10']:.1%} recall@10")
        
        # Balanced recommendation
        balanced_candidates = [s for s in config_summaries if s['recall_10'] >= 0.8]
        if balanced_candidates:
            balanced = min(balanced_candidates, key=lambda x: x['query_time_ms'])
            print(f"\n⚖️  Balanced Recommendation (80%+ recall):")
            print(f"   Configuration: {balanced['config_str']}")
            print(f"   Optimal n_probe: {balanced['best_n_probe']}")
            print(f"   Performance: {balanced['recall_10']:.1%} recall@10, {balanced['query_time_ms']:.2f}ms")
        
        # Analysis insights
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 30)
        
        avg_parents = np.mean([s['num_parents'] for s in config_summaries])
        max_parents = max([s['num_parents'] for s in config_summaries])
        min_parents = min([s['num_parents'] for s in config_summaries])
        
        print(f"📈 Parent Distribution at Level 2:")
        print(f"   Range: {min_parents:,} to {max_parents:,} parents")
        print(f"   Average: {avg_parents:,.0f} parents")
        print(f"   Typical max n_probe: {int(0.5 * avg_parents)} (constraint)")
        
        best_recalls = [s['recall_10'] for s in by_recall[:3]]
        print(f"\n🎯 Recall Performance:")
        print(f"   Best recall@10: {max(best_recalls):.1%}")
        print(f"   Top 3 average: {np.mean(best_recalls):.1%}")
        
        best_speeds = [s['query_time_ms'] for s in by_speed[:3]]
        print(f"\n⚡ Speed Performance:")
        print(f"   Fastest query: {min(best_speeds):.2f}ms")
        print(f"   Top 3 average: {np.mean(best_speeds):.2f}ms")
    
    def save_results(self, results: List[Dict]):
        """Save detailed results to JSON file."""
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_info': {
                'dataset': '1M SIFT',
                'constraints': {
                    'parent_level': 2,
                    'n_probe_constraint': '≤ 0.5 × num_parents'
                },
                'test_queries': self.test_queries,
                'base_vectors': len(self.base_vectors) if self.base_vectors is not None else 0
            },
            'results': results,
            'summary': {
                'total_configurations': len(results),
                'successful_tests': len(results)
            }
        }
        
        filename = 'sift_1m_level2_parameter_sweep.json'
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {filename}")
        print(f"💾 File size: ~{os.path.getsize(filename) / 1024:.1f} KB")

def main():
    """Main execution function."""
    print("🚀 1M SIFT Dataset Parameter Testing")
    print("Constraints: parent_level=2, n_probe ≤ 0.5 × num_parents")
    print("=" * 70)
    
    # Check dataset availability
    required_files = [
        "sift/sift_base.fvecs",
        "sift/sift_query.fvecs", 
        "sift/sift_groundtruth.ivecs"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required SIFT dataset files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure the complete SIFT dataset is available.")
        return
    
    # Initialize tester
    tester = SIFT1MParameterTester(test_queries=1000)  # Test with 1K queries
    
    try:
        # Load dataset
        tester.load_dataset()
        
        # Run parameter sweep
        tester.run_parameter_sweep()
        
        print("\n✅ 1M Parameter Testing Completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
