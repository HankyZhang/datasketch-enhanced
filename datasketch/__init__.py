"""
DataSketch Enhanced - HNSW专版
===============================

这是专注于HNSW (Hierarchical Navigable Small World) 算法的数据草图库增强版。

主要功能：
- HNSW: 高效的近似最近邻搜索算法
- 支持任意距离函数
- 动态插入、删除和更新
- 完整的中文文档和注释

示例使用：
---------
    >>> from datasketch import HNSW
    >>> import numpy as np
    >>> 
    >>> # 创建HNSW索引
    >>> index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
    >>> 
    >>> # 添加数据
    >>> data = np.random.random((1000, 10))
    >>> index.update({i: vec for i, vec in enumerate(data)})
    >>> 
    >>> # 搜索最近邻
    >>> query = np.random.random(10)
    >>> neighbors = index.query(query, k=10)
"""

from .version import __version__
from .hnsw import HNSW

__all__ = ['HNSW', '__version__']