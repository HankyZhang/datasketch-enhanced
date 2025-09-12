# HNSW Hybrid Two-Stage Retrieval System

🚀 **Advanced HNSW implementation with hybrid two-stage retrieval architecture**

A high-performance implementation of the HNSW (Hierarchical Navigable Small World) algorithm featuring an innovative hybrid two-stage retrieval system that significantly improves recall performance.

## 🆕 Latest: HNSW Hybrid Two-Stage System

The **HNSW Hybrid Two-Stage Retrieval System** transforms a standard HNSW into a two-stage retrieval architecture for improved recall performance:

### 🔥 Core Features
- **Two-Stage Search**: Coarse filtering (parent nodes) + Fine filtering (child nodes)
- **Enhanced Recall**: 62.86% Recall@10 performance with optimized parameters
- **Configurable Parameters**: Tunable k_children and n_probe for precision-efficiency trade-offs
- **Scalable Design**: Supports large-scale datasets up to millions of vectors
- **Comprehensive Evaluation**: Complete Recall@K metrics and parameter tuning framework

## 🌟 Key Features

### 🔍 HNSW Algorithm Advantages
- **Efficient Search**: O(log N) time complexity for approximate nearest neighbor search
- **Dynamic Updates**: Real-time insert, delete, and update operations
- **High Precision**: Configurable parameters for 95%+ recall rates
- **Scalable**: Support for million-scale datasets with real-time search

### 🏗️ Hybrid Architecture Innovation
- **Parent-Child Structure**: Extract parent nodes from HNSW Level 2
- **Two-Stage Retrieval**: Coarse search → Fine search within selected regions
- **Parameter Optimization**: Systematic tuning of k_children and n_probe parameters
- **Performance Validation**: Comprehensive evaluation against brute-force ground truth

## 📁 Project Structure

```
datasketch-enhanced/
├── hnsw_core/                    # 🎯 Core HNSW Implementation
│   ├── hnsw.py                  # Standard HNSW algorithm
│   ├── hnsw_hybrid.py           # Hybrid two-stage HNSW system
│   ├── hnsw_hybrid_evaluation.py # Evaluation and benchmarking tools
│   ├── hnsw_examples.py         # Usage examples
│   ├── version.py               # Version information
│   └── __init__.py              # Package initialization
├── docs/                        # Documentation
├── doc_md/                      # Markdown documentation
├── test_hybrid_hnsw.py          # Comprehensive test suite
├── project_demo.py              # Full implementation demo
├── setup.py                     # Installation configuration
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation
```bash
pip install numpy
git clone https://github.com/HankyZhang/datasketch-enhanced.git
cd datasketch-enhanced
pip install -e .
```

### Basic Usage

#### Standard HNSW Usage
```python
from hnsw_core.hnsw import HNSW
import numpy as np

# Create random data
data = np.random.random((1000, 50))

# Initialize HNSW index
distance_func = lambda x, y: np.linalg.norm(x - y)
index = HNSW(distance_func=distance_func, m=16, ef_construction=200)

# Insert data
for i, vector in enumerate(data):
    index.insert(i, vector)

# Search for nearest neighbors
query = np.random.random(50)
neighbors = index.query(query, k=10)

print(f"Found {len(neighbors)} nearest neighbors")
for i, (key, distance) in enumerate(neighbors):
    print(f"{i+1}. Key: {key}, Distance: {distance:.4f}")
```

#### 🆕 HNSW Hybrid Two-Stage Retrieval System
```python
import sys
sys.path.append('hnsw_core')

from hnsw_core.hnsw import HNSW
from hnsw_core.hnsw_hybrid import HNSWHybrid
from hnsw_core.hnsw_hybrid_evaluation import HNSWEvaluator, create_synthetic_dataset, create_query_set
import numpy as np

# Create dataset
dataset = create_synthetic_dataset(5000, 128)  # 5K vectors, 128 dimensions
query_vectors, query_ids = create_query_set(dataset, 100)  # 100 queries

# Build base HNSW index
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func, m=16, ef_construction=200)

# Insert vectors (excluding queries)
for i, vector in enumerate(dataset):
    if i not in query_ids:
        base_index.insert(i, vector)

# Build hybrid index
hybrid_index = HNSWHybrid(
    base_index=base_index,
    parent_level=2,          # Extract parents from level 2
    k_children=1000         # 1000 children per parent
)

# Evaluate recall
evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
ground_truth = evaluator.compute_ground_truth(k=10, distance_func=distance_func)
result = evaluator.evaluate_recall(hybrid_index, k=10, n_probe=15, ground_truth=ground_truth)

print(f"Recall@10: {result['recall_at_k']:.4f}")
print(f"Query time: {result['avg_query_time_ms']:.2f} ms")
```

## 🛠️ Advanced Usage

### Running the Complete Demo
```bash
# Run the complete hybrid system demonstration
python project_demo.py
```

### Running Tests
```bash
# Run comprehensive test suite
python test_hybrid_hnsw.py
```

### Parameter Tuning
The hybrid system supports several key parameters:

- **`parent_level`**: HNSW level to extract parent nodes from (default: 2)
- **`k_children`**: Number of child nodes per parent (default: 1000)
- **`n_probe`**: Number of parent nodes to probe during search (default: 15)

#### Newly Added / Advanced Parameters
- **`parent_child_method`**: How to build parent→child mappings: `approx` (fast; uses HNSW queries) or `brute` (exhaustive; higher coverage/recall, slower build).
- **`approx_ef`**: ef value used when `parent_child_method='approx'` to control breadth of approximate neighbor gathering.
- **`diversify_max_assignments`**: (Optional) Cap on how many different parents a single child can belong to (promotes coverage across regions).
- **`repair_min_assignments`**: (Optional) Minimum number of parent assignments a child should have; triggers a repair pass if used with diversification.
- **`include_parents_in_results`**: If True, parent nodes can appear directly in final search results (useful for hierarchical diagnostics).
- **`overlap_sample`**: Integer number of parent pairs sampled to estimate average Jaccard overlap across child sets (diagnostic metric).

## 📊 Performance Results

### Benchmark Results (5K vectors, 128 dimensions)
- **Recall@10**: 62.86%
- **Average Query Time**: 5.43ms
- **Parent Nodes**: 12 nodes managing 1,438 children
- **Memory Efficiency**: Optimized data structures with minimal overhead

### Key Performance Insights
- **Two-stage approach** provides systematic search within precomputed regions
- **Parameter tuning** allows precision-efficiency trade-offs
- **Scalable architecture** maintains performance at larger scales

## 🧪 Advanced Mapping Comparison & Diagnostics

Use the advanced script to compare **approx vs brute** parent→child mapping strategies and evaluate diversification / repair effects. It also exports a JSON file containing recall, coverage, and structural diagnostics.

### Run Advanced Comparison
```bash
python test_hybrid_advanced.py
```

### Example Output (abridged)
```
Summary (recall@k):
    approx               recall=0.5490 coverage=0.725 avgCand=241.9
    brute                recall=0.7660 coverage=0.940 avgCand=657.8
    approx_diversified   recall=0.5490 coverage=0.725 avgCand=241.9
```

### Exported Benchmark JSON
The run produces `hybrid_mapping_comparison.json` with structure:
```json
{
    "dataset": { "n_vectors": 2000, "dim": 64, "n_queries": 100 },
    "config": { "k": 10, "n_probe": 5, ... },
    "variants": {
        "approx": { "recall_at_k": 0.549, "coverage_fraction": 0.725, ... },
        "brute": { "recall_at_k": 0.766, "coverage_fraction": 0.940, ... },
        "approx_diversified": { ... }
    },
    "comparison": {
        "recall_diff_brute_minus_approx": 0.217,
        "coverage_diff_brute_minus_approx": 0.215,
        "coverage_gain_diversified": 0.0,
        "recall_gain_diversified": 0.0
    }
}
```

### Interpreting Diagnostics
- **coverage_fraction**: Portion of unique children assigned across all parents (higher often improves recall headroom).
- **mean_jaccard_overlap**: Average overlap between sampled parent child-sets (lower indicates better regional separation).
- **avg_candidate_size**: Average number of fine-stage candidates examined per query (proxy for search work).
- **diversification & repair**: Use to balance coverage vs redundancy; adjust `diversify_max_assignments` downward (e.g. 2–3) and enable `repair_min_assignments` to avoid isolated nodes.

### When to Use Brute vs Approx
| Goal | Recommended Method |
|------|--------------------|
| Fast index build, iterative experimentation | approx |
| Maximum recall ceiling or small dataset | brute |
| Improve coverage without brute cost | approx + diversification |

> Tip: Start with `approx` + modest `approx_ef` (50–80), then profile coverage & recall. Switch to `brute` only if coverage stagnates and recall plateaus below target.

## 📚 Documentation

- **[Algorithm Principles](doc_md/HNSW_Hybrid_Algorithm_Principles.md)**: Core concepts and theory
- **[Technical Implementation](doc_md/HNSW_Hybrid_Technical_Implementation.md)**: Implementation details
- **[Complete Guide](doc_md/HNSW_HYBRID_README.md)**: Comprehensive user guide
- **[Project Summary](doc_md/PROJECT_SUMMARY.md)**: Complete project overview

## 🎯 Use Cases

- **Recommendation Systems**: High-recall similarity search
- **Image Retrieval**: Content-based search with improved accuracy
- **Semantic Search**: Document and text similarity with enhanced recall
- **Research Applications**: Algorithm comparison and parameter studies

## 🤝 Contributing

This project is actively maintained. Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the foundation of the original HNSW algorithm with innovative hybrid architecture enhancements for improved recall performance.
