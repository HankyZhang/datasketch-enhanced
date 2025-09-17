# Method 3: K-Means HNSW Parameter Tuning Guide

## Overview of `tune_kmeans_hnsw.py`

This file is the **parameter tuning and evaluation framework** for Method 3 (K-Means HNSW). It provides comprehensive tools to:

1. **Evaluate performance** of K-Means HNSW systems
2. **Optimize parameters** through systematic sweeps
3. **Compare with baseline** HNSW performance
4. **Generate detailed reports** with metrics and timing

## Main Classes and Functions

### 1. `KMeansHNSWEvaluator` Class

The core evaluation class with these key methods:

```python
# Initialize evaluator
evaluator = KMeansHNSWEvaluator(dataset, query_vectors, query_ids, distance_func)

# Compute ground truth (brute force search)
ground_truth = evaluator.compute_ground_truth(k=10)

# Evaluate recall performance
results = evaluator.evaluate_recall(kmeans_hnsw, k=10, n_probe=10)

# Parameter sweep optimization
sweep_results = evaluator.parameter_sweep(base_index, param_grid, eval_params)

# Find optimal parameters
optimal = evaluator.find_optimal_parameters(sweep_results, 'recall_at_k')

# Compare with baseline HNSW
comparison = evaluator.compare_with_baselines(kmeans_hnsw, base_index)
```

### 2. Utility Functions

- `save_results()`: Save results to JSON file
- `load_sift_data()`: Load SIFT dataset if available

## Key Parameters to Adjust

### 1. **Test Data Size** (Most Important)

You can change the test data size in several places:

#### **Change Data Size Here:**

**Location 1 - Synthetic Data Size** (lines 463-464):
```python
# Current: 10K base vectors, 100 queries
base_vectors = np.random.randn(10000, 128).astype(np.float32)  # Change 10000
query_vectors = np.random.randn(100, 128).astype(np.float32)   # Change 100
```

**Location 2 - Query Subset** (line 467):
```python
query_vectors = query_vectors[:100]  # Change 100 to desired number
```

**Location 3 - SIFT Data Subset** (you can add after SIFT loading):
```python
# Add this after loading SIFT data to use subset
base_vectors = base_vectors[:50000]  # Use first 50K instead of full 1M
query_vectors = query_vectors[:200]  # Use 200 queries instead of 10K
```

### 2. **Parameter Grid for Optimization**

#### **Adjustable Parameters:**

**K-Means HNSW System Parameters:**
```python
param_grid = {
    'n_clusters': [20, 50, 100],           # Number of K-Means clusters
    'k_children': [500, 1000, 2000],       # Children per cluster
    'child_search_ef': [100, 200, 400]     # HNSW search width
}
```

**Evaluation Parameters:**
```python
evaluation_params = {
    'k_values': [10],                      # Recall@k values to test
    'n_probe_values': [5, 10, 20]         # Number of clusters to probe
}
```

### 3. **Number of Parameter Combinations**

Change `max_combinations=9` to test more combinations (current grid has 3×3×3 = 27 total combinations).

## How to Use the File

### **Method 1: Run as Script**
```bash
cd method3
python tune_kmeans_hnsw.py
```

### **Method 2: Customize and Run**

Use the provided `custom_tuning.py` file with these configuration options:

```python
# ========== CONFIGURATION ==========
# Data size settings
DATASET_SIZE = 5000      # Number of base vectors (change this!)
QUERY_SIZE = 50          # Number of queries (change this!)
DIMENSION = 128          # Vector dimension

# Parameter grid (adjust these!)
PARAM_GRID = {
    'n_clusters': [10, 25, 50],           # K-Means clusters
    'k_children': [200, 500, 1000],       # Children per cluster  
    'child_search_ef': [50, 100, 200]     # HNSW search width
}

EVAL_PARAMS = {
    'k_values': [10],                     # Recall@k
    'n_probe_values': [3, 5, 10]         # Clusters to probe
}

MAX_COMBINATIONS = 12    # Test 12 out of 27 combinations
USE_SIFT_DATA = False    # Set True to use SIFT, False for synthetic
```

## Parameter Recommendations

### **For Different Dataset Sizes:**

| Dataset Size | n_clusters | k_children | Queries | Combinations |
|-------------|------------|------------|---------|--------------|
| 1K-5K       | [5, 10, 20] | [100, 200, 500] | 20-50 | 9-12 |
| 10K-50K     | [20, 50, 100] | [500, 1000, 2000] | 50-100 | 12-18 |
| 100K-500K   | [50, 100, 200] | [1000, 2000, 5000] | 100-200 | 18-27 |
| 1M+         | [100, 200, 500] | [2000, 5000, 10000] | 200-500 | 27+ |

### **Key Guidelines:**

1. **n_clusters**: ~1-5% of dataset size
2. **k_children**: 10-50x n_clusters 
3. **child_search_ef**: 1-4x k_children
4. **n_probe**: 10-50% of n_clusters

## Quick Usage Examples

### **Small Test (Fast)**
```python
# In custom_tuning.py, change:
DATASET_SIZE = 1000
QUERY_SIZE = 20
PARAM_GRID = {
    'n_clusters': [5, 10],
    'k_children': [100, 200], 
    'child_search_ef': [50, 100]
}
MAX_COMBINATIONS = 4
```

### **Medium Evaluation**
```python
DATASET_SIZE = 10000
QUERY_SIZE = 100
PARAM_GRID = {
    'n_clusters': [20, 50, 100],
    'k_children': [500, 1000],
    'child_search_ef': [100, 200, 400]
}
MAX_COMBINATIONS = 12
```

### **Full SIFT Evaluation**
```python
USE_SIFT_DATA = True
DATASET_SIZE = 100000  # Use 100K from SIFT 1M
QUERY_SIZE = 500
MAX_COMBINATIONS = 18
```

## Output and Results

The script generates:
1. **Console output** with progress and results
2. **JSON file** with detailed metrics
3. **Optimal parameters** for your dataset
4. **Baseline comparison** showing performance vs standard HNSW

## Main Functions Explained

### `compute_ground_truth(k, exclude_query_ids=True)`
- Computes true nearest neighbors using brute force search
- Used as reference for recall calculation
- `k`: number of neighbors to find
- `exclude_query_ids`: whether to exclude query points from results

### `evaluate_recall(kmeans_hnsw, k, n_probe, ground_truth=None)`
- Evaluates recall@k performance of K-Means HNSW system
- Compares search results with ground truth
- Returns detailed metrics including timing

### `parameter_sweep(base_index, param_grid, evaluation_params, max_combinations=None)`
- Tests multiple parameter combinations systematically
- Builds K-Means HNSW system for each combination
- Evaluates performance across different settings

### `find_optimal_parameters(sweep_results, optimization_target='recall_at_k', constraints=None)`
- Finds best parameters from sweep results
- Can optimize for recall, speed, or other metrics
- Supports constraints (e.g., max query time)

### `compare_with_baselines(kmeans_hnsw, base_index, k=10, n_probe=10, ef_values=None)`
- Compares K-Means HNSW with standard HNSW
- Tests different ef values for baseline
- Provides performance comparison
