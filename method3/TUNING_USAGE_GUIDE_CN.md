# K-Means HNSW参数调优工具使用指南

## 📚 概述

`tune_kmeans_hnsw.py` 是一个全面的参数调优和评估工具，用于优化K-Means HNSW(层次化可导航小世界图)混合搜索系统的性能。该工具提供系统性的参数扫描、性能评估和基准对比功能。

## 🎯 主要功能

### 1. 算法对比评估
- **基线HNSW**: 原始HNSW算法性能基准
- **纯K-Means聚类**: MiniBatchKMeans聚类搜索
- **K-Means HNSW混合**: 两阶段混合搜索系统

### 2. 参数自动调优
- 聚类数量(n_clusters)优化
- 子节点数量(k_children)调整
- 搜索效率参数(child_search_ef)优化
- 探测聚类数(n_probe)配置

### 3. 性能指标分析
- **召回率(Recall@K)**: 算法准确性评估
- **查询时间**: 平均查询延迟分析
- **构建时间**: 索引构建耗时统计
- **内存使用**: 系统资源消耗监控

## 🚀 快速开始

### 基本使用
```bash
# 使用默认参数运行(10K数据集，50个查询)
python tune_kmeans_hnsw.py

# 使用SIFT数据集
python tune_kmeans_hnsw.py --dataset-size 50000 --query-size 1000

# 强制使用合成数据
python tune_kmeans_hnsw.py --no-sift --dataset-size 20000 --dimension 256
```

### 高级配置
```bash
# 启用自适应参数调整
python tune_kmeans_hnsw.py \
    --adaptive-k-children \
    --k-children-scale 2.0 \
    --k-children-min 150 \
    --k-children-max 500

# 启用多样化分配和修复机制
python tune_kmeans_hnsw.py \
    --diversify-max-assignments 10 \
    --repair-min-assignments 3 \
    --manual-repair \
    --manual-repair-min 2
```

## 📊 参数详解

### 数据集参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-size` | 10000 | 基础向量数量，影响搜索准确性和构建时间 |
| `--query-size` | 50 | 查询向量数量，用于性能评估 |
| `--dimension` | 128 | 向量维度(仅用于合成数据) |
| `--no-sift` | False | 强制使用合成数据而非SIFT数据集 |

### 自适应参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--adaptive-k-children` | False | 启用基于平均聚类大小的自适应子节点数 |
| `--k-children-scale` | 1.5 | 自适应缩放因子，影响子节点数量计算 |
| `--k-children-min` | 100 | 自适应模式下的最小子节点数 |
| `--k-children-max` | None | 自适应模式下的最大子节点数(可选) |

### 优化策略参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--diversify-max-assignments` | None | 每个子节点的最大分配数，启用多样化策略 |
| `--repair-min-assignments` | None | 构建期间的最小分配数，启用修复机制 |
| `--manual-repair` | False | 在最优构建后执行手动修复 |
| `--manual-repair-min` | None | 手动修复的最小分配阈值 |

## 🔬 核心算法原理

### K-Means HNSW混合搜索
1. **第一阶段**: 使用K-Means聚类快速定位候选区域
2. **第二阶段**: 在选定聚类内使用HNSW进行精确搜索
3. **优势**: 结合了聚类的粗粒度过滤和HNSW的高精度搜索

### 参数调优策略
```python
# 自动参数网格定义
param_grid = {
    'n_clusters': [32, 64, 128],      # 聚类数量选择
    'k_children': [200, 400],         # 子节点数量
    'child_search_ef': [300, 500]     # 搜索效率参数
}

evaluation_params = {
    'k_values': [10],                 # 评估K值
    'n_probe_values': [5, 10, 20]     # 探测聚类数
}
```

## 📈 输出结果解读

### 性能指标
```
🎯 算法对比结果:
K-Means HNSW: Recall=0.8500, Time=12.34ms
Pure K-Means: Recall=0.7200, Time=8.76ms  
Best Baseline: Recall=0.9100, Time=45.67ms
```

- **Recall**: 召回率，值越高表示找到的真实邻居越多
- **Time**: 平均查询时间，值越低表示搜索速度越快
- **权衡**: 通常召回率和速度之间需要权衡

### 详细分析输出
```
📊 详细纯K-Means结果:
  Overall Recall@10: 0.7200
  Average Individual Recall: 0.7150
  Correct/Expected: 360/500
  Clustering Time: 2.45s
  Average Query Time: 8.76ms
```

## 🛠️ 常见使用场景

### 场景1: 快速原型验证
```bash
# 小规模快速测试
python tune_kmeans_hnsw.py --dataset-size 5000 --query-size 20
```

### 场景2: 生产环境调优
```bash
# 大规模性能优化
python tune_kmeans_hnsw.py \
    --dataset-size 100000 \
    --query-size 500 \
    --adaptive-k-children \
    --diversify-max-assignments 8
```

### 场景3: 算法研究分析
```bash
# 详细性能分析
python tune_kmeans_hnsw.py \
    --no-sift \
    --dimension 512 \
    --manual-repair \
    --repair-min-assignments 5
```

## 📁 输出文件

### method3_tuning_results.json
调优完成后，结果会保存到JSON文件中，包含：
- `sweep_results`: 所有参数组合的详细结果
- `optimal_parameters`: 最优参数配置
- `baseline_comparison`: 与基线算法的对比
- `evaluation_info`: 评估环境信息

## 🔧 自定义扩展

### 添加新的参数
```python
# 在param_grid中添加新参数
param_grid = {
    'n_clusters': [16, 32, 64],
    'k_children': [100, 200, 400],
    'child_search_ef': [200, 300, 500],
    'custom_param': [1, 2, 3]  # 新参数
}
```

### 自定义评估指标
```python
# 添加新的性能约束
constraints = {
    'avg_query_time_ms': 50.0,    # 最大查询时间
    'recall_at_k': 0.8            # 最小召回率
}
```

## ⚡ 性能优化建议

### 数据集大小选择
- **小数据集(<10K)**: 使用较少聚类数(10-32)
- **中等数据集(10K-100K)**: 使用适中聚类数(32-128)  
- **大数据集(>100K)**: 使用更多聚类数(128-512)

### 参数调优策略
1. **先粗调后细调**: 先用大步长找到大致范围，再细化
2. **权衡召回率和速度**: 根据应用需求调整参数优先级
3. **使用自适应参数**: 对于变化的数据分布，启用自适应机制

### 内存优化
- 使用MiniBatchKMeans减少内存占用
- 限制k_children数量避免内存爆炸
- 启用多样化分配机制提高内存效率

## 🐛 故障排除

### 常见问题

**Q: SIFT数据加载失败**
```
Error loading SIFT data: FileNotFoundError
Using synthetic data instead...
```
A: 确保sift目录下有正确的.fvecs文件，或使用`--no-sift`强制使用合成数据

**Q: 内存不足错误**
```
MemoryError: Unable to allocate array
```
A: 减少`--dataset-size`或启用`--diversify-max-assignments`限制内存使用

**Q: 参数调优时间过长**
```
Testing 27 parameter combinations...
```
A: 减少参数网格大小或设置较小的`max_combinations`

### 调试技巧
1. 使用小数据集快速验证参数
2. 检查JSON输出文件中的详细错误信息
3. 启用详细日志输出观察调优过程

## 📚 相关资料

- [HNSW算法原理](../docs/HNSW算法原理详解.md)
- [K-Means聚类理论](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [SIFT特征描述符](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
- [向量相似性搜索综述](https://arxiv.org/abs/1603.09320)

---

**最后更新**: 2025年9月
**版本**: 1.0
**作者**: datasketch-enhanced团队
