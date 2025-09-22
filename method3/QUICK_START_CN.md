# K-Means HNSW参数调优工具 - 快速上手指南

## 🚀 快速开始

这是一个用于优化K-Means HNSW混合搜索系统性能的参数调优工具。

### 基本使用命令

```bash
# 1. 基本运行 - 使用默认参数
python method3/tune_kmeans_hnsw.py

# 2. 自定义数据集大小
python method3/tune_kmeans_hnsw.py --dataset-size 20000 --query-size 100

# 3. 使用合成数据（不使用SIFT数据集）
python method3/tune_kmeans_hnsw.py --no-sift --dimension 256

# 4. 启用自适应参数调整
python method3/tune_kmeans_hnsw.py --adaptive-k-children --k-children-scale 2.0
```

## 📊 主要功能

### 1. 算法性能对比
- **基线HNSW**: 原始层次化可导航小世界图算法
- **纯K-Means**: MiniBatchKMeans聚类搜索
- **K-Means HNSW**: 两阶段混合搜索系统

### 2. 自动参数优化
自动测试不同参数组合，找到最优配置：
- 聚类数量 (n_clusters)
- 子节点数量 (k_children) 
- 搜索效率参数 (child_search_ef)

### 3. 性能评估指标
- **召回率 (Recall@K)**: 算法准确性，越高越好
- **查询时间**: 平均查询延迟，越低越好
- **构建时间**: 索引构建耗时

## 🎯 常用参数组合

### 快速测试（适合开发调试）
```bash
python method3/tune_kmeans_hnsw.py \
    --dataset-size 5000 \
    --query-size 20 \
    --no-sift
```

### 标准评估（适合性能测试）
```bash
python method3/tune_kmeans_hnsw.py \
    --dataset-size 50000 \
    --query-size 1000
```

### 高级优化（适合生产调优）
```bash
python method3/tune_kmeans_hnsw.py \
    --dataset-size 100000 \
    --query-size 500 \
    --adaptive-k-children \
    --k-children-scale 1.8 \
    --diversify-max-assignments 10 \
    --manual-repair
```

## 📈 结果解读

运行完成后，你会看到类似以下的输出：

```
🎯 算法对比结果:
K-Means HNSW: Recall=0.8500, Time=12.34ms
Pure K-Means: Recall=0.7200, Time=8.76ms  
Best Baseline: Recall=0.9100, Time=45.67ms
```

**解读说明**:
- **K-Means HNSW**: 混合算法，平衡了速度和准确性
- **Pure K-Means**: 最快但准确性较低
- **Best Baseline**: 最准确但速度较慢

## 📁 输出文件

调优完成后会生成 `method3_tuning_results.json` 文件，包含：
- 所有测试参数的详细结果
- 最优参数配置建议
- 性能对比数据

## ⚙️ 关键参数说明

| 参数名 | 默认值 | 说明 | 推荐设置 |
|--------|--------|------|----------|
| `--dataset-size` | 10000 | 基础数据集大小 | 5000(测试) / 50000(评估) |
| `--query-size` | 50 | 查询数量 | 20(测试) / 500(评估) |
| `--adaptive-k-children` | False | 启用自适应子节点数 | 推荐启用 |
| `--k-children-scale` | 1.5 | 自适应缩放因子 | 1.5-2.0 |
| `--no-sift` | False | 强制使用合成数据 | 测试时启用 |

## 🛠️ 故障排除

### 问题1: SIFT数据加载失败
```
Error loading SIFT data: FileNotFoundError
Using synthetic data instead...
```
**解决方案**: 添加 `--no-sift` 参数强制使用合成数据

### 问题2: 内存不足
```
MemoryError: Unable to allocate array
```
**解决方案**: 减少 `--dataset-size` 参数值

### 问题3: 运行时间过长
**解决方案**: 
- 减少数据集大小
- 使用 `--no-sift` 跳过SIFT数据加载
- 减少查询数量

## 💡 性能优化建议

1. **首次使用**: 先用小数据集测试 (`--dataset-size 5000`)
2. **生产调优**: 逐步增加数据集大小观察性能变化
3. **内存优化**: 启用 `--diversify-max-assignments` 限制内存使用
4. **速度优化**: 调整 `k-children-scale` 在速度和准确性间平衡

## 📞 技术支持

如遇问题，请检查：
1. Python环境是否包含numpy和sklearn
2. 当前目录是否为项目根目录
3. 是否有足够的内存运行所选数据集大小

更多详细信息请参考：[完整使用指南](TUNING_USAGE_GUIDE_CN.md)
