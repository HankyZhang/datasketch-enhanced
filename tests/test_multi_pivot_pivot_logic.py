import numpy as np
import math

from method3 import KMeansHNSWMultiPivot
from hnsw.hnsw import HNSW


def build_base(vectors):
    dist = lambda a, b: float(np.linalg.norm(a - b))
    index = HNSW(distance_func=dist, m=8, ef_construction=64)
    for i, v in enumerate(vectors):
        index.insert(i, v.astype(np.float32))
    return index


def test_multi_pivot_third_is_perp():
    """验证第三枢纽选择为对 AB 直线垂距最大的点 (max_perp_AB)."""
    # 构造一组沿 x 轴 + 一个偏离点，期望 C 选到偏离点
    pts = np.array([
        [-20.0, 0.0],   # id 0  (将成为距质心最远 -> B)
        [5.0, 0.0],     # id 1
        [15.0, 0.0],    # id 2
        [30.0, 0.0],    # id 3
        [15.0, 10.0],   # id 4  (垂直方向远离直线)
    ], dtype=np.float32)
    base = build_base(pts)

    mp = KMeansHNSWMultiPivot(
        base_index=base,
        n_clusters=1,
        k_children=5,
        num_pivots=3,
        pivot_selection_strategy='line_perp_third',
        pivot_overquery_factor=1.0,
        child_search_ef=50,
        store_pivot_debug=True,
    )

    dbg = mp.get_pivot_debug()['centroid_0']
    # 期望 pivot_types = ['centroid','farthest_from_A','max_perp_AB']
    assert dbg['pivot_types'][0] == 'centroid'
    assert dbg['pivot_types'][1] == 'farthest_from_A'
    assert dbg['pivot_types'][2] == 'max_perp_AB'
    # 第三个枢纽对应 id 4
    assert dbg['pivot_ids'][2] == 4, f"Expected C id=4 got {dbg['pivot_ids'][2]}"


def test_multi_pivot_fallback_third():
    """验证当 A 与 B 重合 (v≈0) 时第三枢纽走 fallback_max_dist_A。"""
    # 所有点重合 -> 质心 == 各点；B 距离 0；v=0 -> fallback 逻辑
    pts = np.zeros((4, 2), dtype=np.float32)
    base = build_base(pts)

    mp = KMeansHNSWMultiPivot(
        base_index=base,
        n_clusters=1,
        k_children=4,
        num_pivots=3,
        pivot_selection_strategy='line_perp_third',
        pivot_overquery_factor=1.0,
        child_search_ef=10,
        store_pivot_debug=True,
    )
    dbg = mp.get_pivot_debug()['centroid_0']
    assert dbg['pivot_types'][1] == 'farthest_from_A'
    # 由于 v_norm_sq < eps, 第三个枢纽应 fallback
    assert dbg['pivot_types'][2] == 'fallback_max_dist_A'
