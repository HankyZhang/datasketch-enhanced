# Method 3: K-Means-based Two-Stage HNSW System

This directory implements Method 3 of the enhanced datasketch project - a novel two-stage retrieval system that combines K-Means clustering with HNSW for efficient approximate nearest neighbor search.

## Overview

Method 3 uses a three-phase approach:

1. **Phase 1**: Standard HNSW Index (reuses existing base_index)
2. **Phase 2**: K-Means Parent Discovery + HNSW Child Assignment
3. **Phase 3**: Two-Stage Search (K-Means centroids → HNSW children)

### Key Innovation

Unlike Method 2 which uses HNSW hierarchy levels as parents, Method 3 uses K-Means centroids as parent nodes, providing:
- More balanced cluster sizes
- Better representation of data distribution
- Configurable number of parent nodes independent of HNSW structure
- Improved recall through principled clustering

## Architecture

```
Query Vector
     ↓
Stage 1: Find closest K-Means centroids (fast brute force)
     ↓
Stage 2: Search HNSW-assigned children of selected centroids
     ↓
Top-k Results
```

## File Structure

- `kmeans_hnsw.py` - Core implementation of KMeansHNSW system
- `tune_kmeans_hnsw.py` - Parameter tuning and evaluation framework  
- `example_usage.py` - Comprehensive usage examples and demos
- `__init__.py` - Package initialization and exports
- `README.md` - This documentation

## Quick Start

```python
from method3 import KMeansHNSW
from hnsw.hnsw import HNSW
import numpy as np

# Build base HNSW index
distance_func = lambda x, y: np.linalg.norm(x - y)
base_index = HNSW(distance_func=distance_func)

# Insert your data
for i, vector in enumerate(dataset):
    base_index.insert(i, vector)

# Create K-Means HNSW system
kmeans_hnsw = KMeansHNSW(
    base_index=base_index,
    n_clusters=50,        # Number of K-Means centroids
    k_children=1000       # Children per centroid via HNSW
)

# Search
results = kmeans_hnsw.search(query_vector, k=10, n_probe=5)
```

## Key Parameters

### System Parameters
- `n_clusters`: Number of K-Means centroids (parent nodes)
- `k_children`: Number of child nodes per centroid (via HNSW search)
- `child_search_ef`: Search width for HNSW child finding (auto-computed)

### Search Parameters  
- `k`: Number of results to return
- `n_probe`: Number of centroids to probe in Stage 1

### Advanced Options
- `diversify_max_assignments`: Limit assignments per child for load balancing
- `repair_min_assignments`: Ensure minimum coverage per child
- `include_centroids_in_results`: Include centroids in search results
- `kmeans_params`: Custom K-Means parameters (iterations, initialization, etc.)

## Performance Characteristics

### Advantages
- **Balanced clustering**: K-Means provides more uniform cluster sizes than HNSW levels
- **Principled parent selection**: Centroids represent actual data distribution
- **Configurable complexity**: Independent control of parent count and children per parent
- **Good recall**: Combines clustering quality with HNSW precision

### Trade-offs
- **Construction time**: K-Means clustering adds overhead during building
- **Memory usage**: Stores both centroids and parent-child mappings
- **Parameter sensitivity**: Requires tuning of clustering and search parameters

## Evaluation Framework

The `KMeansHNSWEvaluator` class provides comprehensive evaluation:

```python
from method3 import KMeansHNSWEvaluator

evaluator = KMeansHNSWEvaluator(dataset, queries, query_ids, distance_func)

# Evaluate recall
results = evaluator.evaluate_recall(kmeans_hnsw, k=10, n_probe=10)

# Parameter sweep
sweep_results = evaluator.parameter_sweep(base_index, param_grid, eval_params)

# Find optimal parameters
optimal = evaluator.find_optimal_parameters(sweep_results, 'recall_at_k')

# Compare with baseline
comparison = evaluator.compare_with_baselines(kmeans_hnsw, base_index)
```

## Integration with Existing Framework

Method 3 is fully compatible with the existing evaluation infrastructure:
- Uses same SIFT dataset format as Methods 1 & 2
- Compatible with existing distance functions
- Follows same interface patterns for consistency
- Reuses ground truth computation and recall metrics

## Examples and Demos

Run `example_usage.py` for comprehensive demonstrations:

1. **Synthetic Data Demo**: Basic usage with random vectors
2. **SIFT Evaluation**: Performance on real SIFT dataset  
3. **Parameter Optimization**: Automated parameter tuning
4. **Advanced Features**: Diversification, repair, and detailed statistics

```bash
cd method3
python example_usage.py
```

## Parameter Tuning

Use `tune_kmeans_hnsw.py` for systematic parameter optimization:

```bash
cd method3  
python tune_kmeans_hnsw.py
```

This will:
- Load SIFT data (or generate synthetic if unavailable)
- Run parameter sweep across multiple configurations
- Find optimal parameters for target recall
- Compare with baseline HNSW performance
- Save results to JSON for analysis

## Expected Performance

Based on initial testing:

- **Recall@10**: 0.85-0.95 (depending on parameters)
- **Search time**: 5-50ms per query (dataset dependent)  
- **Construction time**: 2-3x base HNSW (due to K-Means)
- **Memory usage**: ~1.5x base HNSW (centroids + mappings)

## Future Enhancements

Potential improvements for Method 3:
- Hierarchical K-Means for multiple levels
- Online centroid updates for dynamic datasets
- GPU acceleration for large-scale clustering
- Approximate K-Means for faster construction
