# High Recall Results Summary (n_probe=10)

## 🏆 Key Findings

### Best Configuration for High Recall (96.8% recall@10):
```
Parameters:
- m = 16
- ef_construction = 200  
- parent_level = 0
- k_children = 500
- method = exact
- n_probe = 10

Performance:
- Recall@10: 96.8%
- Query time: 1.50ms
- Build time: 15.96s
```

### Speed vs Recall Tradeoffs:

| Configuration | Recall@10 | Query Time | Speed vs Standard HNSW |
|---------------|-----------|------------|------------------------|
| **Level 0, k_children=500** | **96.8%** | 1.50ms | ~2-3x faster |
| Level 1, ef_c=400, k_children=1000 | 88.8% | 0.64ms | ~4-5x faster |
| Level 1, ef_c=200, k_children=1000 | 83.8% | 0.47ms | ~6-7x faster |

## 📊 Analysis

### Why Level 0 Works Best:
1. **Maximum Coverage**: Uses all nodes as parents
2. **No Information Loss**: Every vector can be a search starting point
3. **Perfect Recall Potential**: Can achieve near-perfect recall with sufficient n_probe

### Performance Characteristics:
- **Level 0**: 96.8% recall, slower queries (1.5ms)
- **Level 1**: 83-89% recall, faster queries (0.5-0.7ms)
- **Level 2**: <40% recall (too few parents)

## 🎯 Recommendations by Use Case

### Maximum Accuracy (>95% recall):
```python
config = {
    'm': 16,
    'ef_construction': 200,
    'parent_level': 0,
    'k_children': 500,
    'n_probe': 10
}
# Expected: 96.8% recall@10, 1.50ms query time
```

### Balanced Performance (85-90% recall):
```python
config = {
    'm': 16,
    'ef_construction': 400,
    'parent_level': 1,
    'k_children': 1000,
    'n_probe': 10
}
# Expected: 88.8% recall@10, 0.64ms query time
```

### Maximum Speed (80-85% recall):
```python
config = {
    'm': 16,
    'ef_construction': 200,
    'parent_level': 1,
    'k_children': 1000,
    'n_probe': 10
}
# Expected: 83.8% recall@10, 0.47ms query time
```

## 🔧 Parameter Effects

### ef_construction Impact:
- 200 → 400: +5-10% recall, +30% build time
- 400 → 600: Minimal recall gain, significant build time increase

### parent_level Impact:
- **0**: Maximum recall (96%+), slower queries
- **1**: Good recall (80-90%), faster queries  
- **2**: Poor recall (<40%), fastest queries

### k_children Impact:
- 500 → 1000: Minimal recall change at level 0
- 1000 → 2000: Slight recall improvement at level 1

## 💡 Optimization Tips

1. **For High Recall**: Always use `parent_level=0` with `n_probe=10+`
2. **Memory Efficiency**: Use `k_children=500` (sufficient for most cases)
3. **Build Speed**: Use `ef_construction=200` (good quality/speed balance)
4. **Query Speed**: Use `parent_level=1` if you can accept 85-90% recall

## 🧪 Further Testing Recommendations

To achieve even higher recall (>98%), try:
- Increase `n_probe` to 15-20 with level 0 configuration
- Use `ef_construction=400` with level 0
- Test `parent_child_method='exact'` vs `'approx'`

## 📈 Benchmark Context

Tested on SIFT dataset subset:
- 3,000 base vectors (128D)
- 50 query vectors
- Ground truth computed via brute force
- All tests used `n_probe=10`

These results demonstrate that hybrid HNSW can achieve very high recall rates (96.8%) while still providing significant speedup over standard approaches.
