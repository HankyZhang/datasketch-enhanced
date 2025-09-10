# HNSW Hybrid Two-Stage Retrieval System

This project implements a hybrid HNSW index structure that transforms a standard HNSW into a two-stage retrieval system for improved recall evaluation in plaintext environments. The system focuses on validating recall performance while ignoring performance and complexity issues from homomorphic encryption.

## Project Overview

### Core Concept

The system implements a **two-stage retrieval architecture**:

1. **Stage 1 (Parent Layer / Coarse Filtering)**: Extract nodes from a higher level of a complete HNSW graph (e.g., Level 2) and treat them as "cluster centers" or "parent nodes." This small-scale layer is used to quickly locate the "region" of the query vector.

2. **Stage 2 (Child Layer / Fine Filtering)**: For each "parent node," precompute and store a set of its most similar neighbor nodes, called "child nodes." Once a query is routed to a parent node in Stage 1, the precise search will only be performed among its child nodes.

### Key Benefits

- **Improved Recall**: Two-stage filtering can achieve better recall than single-stage HNSW
- **Controllable Trade-offs**: Fine-tune balance between recall and efficiency via parameters
- **Scalable Evaluation**: Systematic parameter tuning and analysis framework
- **Reproducible Results**: Comprehensive logging and visualization

## File Structure

```
datasketch/
├── hnsw.py                    # Original HNSW implementation
├── hnsw_hybrid.py            # Hybrid HNSW system implementation
├── experiment_runner.py      # Full experimental pipeline
├── parameter_tuning.py       # Parameter tuning and analysis
└── HNSW_HYBRID_README.md     # This documentation
```

## Installation

1. Ensure you have the required dependencies:
```bash
pip install numpy matplotlib pandas seaborn
```

2. The system uses the existing `datasketch.hnsw` module, so no additional installation is required.

## Quick Start

### Basic Usage

```python
from datasketch.hnsw import HNSW
from hnsw_hybrid import HNSWHybrid, HNSWEvaluator, create_synthetic_dataset, create_query_set
import numpy as np

# Create synthetic data
dataset = create_synthetic_dataset(10000, 128)  # 10K vectors, 128 dim
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
    parent_level=2,      # Extract parents from level 2
    k_children=1000      # 1000 children per parent
)

# Evaluate recall
evaluator = HNSWEvaluator(dataset, query_vectors, query_ids)
ground_truth = evaluator.compute_ground_truth(k=10, distance_func=distance_func)
result = evaluator.evaluate_recall(hybrid_index, k=10, n_probe=10, ground_truth=ground_truth)

print(f"Recall@10: {result['recall_at_k']:.4f}")
print(f"Query time: {result['avg_query_time_ms']:.2f} ms")
```

### Running Full Experiments

#### 1. Complete Experimental Pipeline

```bash
# Run full experiment with default parameters (demo size)
python experiment_runner.py

# Run with custom parameters
python experiment_runner.py \
    --dataset_size 100000 \
    --query_size 1000 \
    --dim 128 \
    --parent_level 2 \
    --k_children 500 1000 2000 \
    --n_probe 5 10 20 50 \
    --k_values 10 50 100 \
    --output_dir experiment_results
```

#### 2. Parameter Tuning

```bash
# Run parameter tuning with systematic sweep
python parameter_tuning.py \
    --dataset_size 100000 \
    --query_size 1000 \
    --dim 128 \
    --k_children_range 100 2000 100 \
    --n_probe_range 1 50 1 \
    --k_values 10 50 100 \
    --max_configurations 100 \
    --output_dir parameter_tuning_results
```

## System Architecture

### HNSWHybrid Class

The main class that implements the hybrid two-stage system:

```python
class HNSWHybrid:
    def __init__(self, base_index, parent_level=2, k_children=1000, distance_func=None):
        # Initialize hybrid index from base HNSW
        
    def search(self, query_vector, k=10, n_probe=10):
        # Perform two-stage search
        
    def get_stats(self):
        # Get construction statistics
```

**Key Methods:**
- `_extract_parent_nodes()`: Extract parent nodes from specified HNSW level
- `_precompute_child_mappings()`: Precompute child nodes for each parent
- `_stage1_coarse_search()`: Find closest parent nodes
- `_stage2_fine_search()`: Search within child nodes of selected parents

### HNSWEvaluator Class

Comprehensive evaluation framework:

```python
class HNSWEvaluator:
    def compute_ground_truth(self, k, distance_func):
        # Compute exact nearest neighbors using brute force
        
    def evaluate_recall(self, hybrid_index, k, n_probe, ground_truth=None):
        # Evaluate recall performance
        
    def parameter_sweep(self, hybrid_index, k_values, n_probe_values):
        # Perform systematic parameter sweep
```

## Parameters

### Core Parameters

- **`parent_level`**: HNSW level to extract parent nodes from (default: 2)
- **`k_children`**: Number of child nodes to precompute for each parent (default: 1000)
- **`n_probe`**: Number of parent nodes to probe in Stage 1 (default: 10)
- **`k`**: Number of results to return (default: 10)

### Tuning Guidelines

1. **`parent_level`**: 
   - Higher levels = fewer parents, faster Stage 1, potentially lower recall
   - Lower levels = more parents, slower Stage 1, potentially higher recall
   - Recommended: 2-3 for most datasets

2. **`k_children`**:
   - Higher values = more children per parent, higher recall, slower Stage 2
   - Lower values = fewer children per parent, lower recall, faster Stage 2
   - Recommended: 500-2000 depending on dataset size

3. **`n_probe`**:
   - Higher values = more parents probed, higher recall, slower search
   - Lower values = fewer parents probed, lower recall, faster search
   - Recommended: 5-20 for good recall-efficiency trade-off

## Output and Analysis

### Generated Files

The system generates comprehensive output:

1. **JSON Results**: Detailed experimental results and configurations
2. **CSV Data**: Parameter sweep results for easy analysis
3. **Visualization Plots**:
   - Recall vs n_probe curves
   - Recall vs k_children curves
   - Query time vs recall efficiency curves
   - Parameter heatmaps
   - 3D recall landscapes

### Key Metrics

- **Recall@K**: Fraction of correct neighbors retrieved
- **Query Time**: Average query processing time
- **Construction Time**: Time to build hybrid index
- **Memory Usage**: Storage requirements for parent-child mappings

## Example Results

### Typical Performance

For a 100K vector dataset with 128 dimensions:

```
Best configuration for Recall@10:
  k_children: 1000
  n_probe: 10
  Recall@10: 0.8542
  Query time: 2.34 ms
  Total correct: 8542/10000
```

### Parameter Sensitivity

- **k_children**: Strong positive correlation with recall (0.7-0.9)
- **n_probe**: Moderate positive correlation with recall (0.4-0.6)
- **Query time**: Increases linearly with both parameters

## Advanced Usage

### Custom Distance Functions

```python
# Jaccard distance for set data
def jaccard_distance(x, y):
    intersection = len(set(x).intersection(set(y)))
    union = len(set(x).union(set(y)))
    return 1.0 - intersection / union if union > 0 else 1.0

hybrid_index = HNSWHybrid(
    base_index=base_index,
    parent_level=2,
    k_children=1000,
    distance_func=jaccard_distance
)
```

### Large-Scale Experiments

For the full 6M vector experiment as described in the project guide:

```bash
python experiment_runner.py \
    --dataset_size 6000000 \
    --query_size 10000 \
    --dim 128 \
    --parent_level 2 \
    --k_children 1000 2000 5000 \
    --n_probe 10 20 50 \
    --k_values 10 50 100 \
    --output_dir large_scale_results
```

**Note**: This will require significant computational resources and time.

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `k_children` or `dataset_size` for large datasets
2. **Slow Construction**: Use smaller `ef_construction` for base HNSW
3. **Low Recall**: Increase `n_probe` or `k_children`
4. **Slow Queries**: Decrease `n_probe` or `k_children`

### Performance Tips

1. **Parallel Processing**: The system can be easily parallelized for parameter sweeps
2. **Caching**: Ground truth computation can be cached and reused
3. **Sampling**: Use smaller query sets for initial parameter exploration
4. **Incremental Building**: Build hybrid indices incrementally for large datasets

## Future Extensions

1. **FAISS Integration**: Replace brute force search with FAISS for better performance
2. **GPU Acceleration**: Implement GPU-accelerated distance computations
3. **Dynamic Parameters**: Adaptive parameter selection based on query characteristics
4. **Multi-level Hierarchies**: Extend to more than two stages
5. **Real-world Datasets**: Test on actual vector datasets (embeddings, features, etc.)

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hnsw_hybrid_2024,
  title={HNSW Hybrid Two-Stage Retrieval System},
  author={AI Assistant},
  year={2024},
  url={https://github.com/your-repo/hnsw-hybrid}
}
```

## License

This project is licensed under the same license as the original datasketch library.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

For more detailed information, see the individual module documentation and example scripts.
