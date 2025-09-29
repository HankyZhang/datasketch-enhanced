"""Unified & simplified K-Means + (Multi) Pivot + HNSW evaluation.

核心思想 (Key Idea)
-------------------
以前的实现里 Single Pivot / Multi Pivot / Hybrid 重复了大量：
1. 向量提取 / 聚类
2. 两阶段检索逻辑（父节点选择 + 子节点候选合并 + 精排）
3. 统计与召回评估

真正的差异只在 “父节点如何选择其子节点” 和（在 multi pivot 情况下）如何生成多个 pivot。
因此这里采用：
  SharedBuild + Strategy Pattern + 统一 TwoStageIndex + 通用 Evaluator

可扩展点：
  - 新增策略只需实现 BaseAssignmentStrategy.assign_children
  - 可插入 repair / diversify 逻辑

依赖：需要已有 HNSW 实现 (hnsw.hnsw.HNSW)；若无可自行替换。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Any, Optional, Hashable, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from hnsw.hnsw import HNSW  # 假设已存在

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_assignment_stats(parent_child: Dict[Hashable, List[Hashable]], base_index: HNSW) -> Dict[str, Any]:
    total = 0
    seen = set()
    dup = 0
    for _, children in parent_child.items():
        total += len(children)
        for c in children:
            if c in seen:
                dup += 1
            else:
                seen.add(c)
    total_base = len(base_index)
    coverage = len(seen) / total_base if total_base else 0.0
    return {
        "total_assignments": total,
        "unique_assigned_nodes": len(seen),
        "duplicate_assignments": dup,
        "duplication_rate": (dup / total) if total else 0.0,
        "coverage_fraction": coverage,
        "total_base_nodes": total_base,
    }


# ------------------------------------------------------------
# 聚类与共享构建 (一次即可复用)
# ------------------------------------------------------------

@dataclass
class SharedContext:
    base_index: HNSW
    node_ids: List[int]
    node_vectors: np.ndarray  # shape (N, D)
    centroids: np.ndarray     # shape (C, D)
    cluster_members: List[List[int]]  # cluster_id -> list[node_id]
    params: Dict[str, Any]


def build_shared(base_index: HNSW, params: Dict[str, Any]) -> SharedContext:
    # 1. 提取所有节点向量
    node_ids: List[int] = []
    vectors: List[np.ndarray] = []
    for nid, node in base_index._nodes.items():  # 类型: _Node
        if node.point is not None:
            node_ids.append(nid)
            vectors.append(node.point)
    if not vectors:
        raise ValueError("No vectors extracted from base_index")
    node_vectors = np.vstack(vectors).astype(np.float32)

    # 2. 聚类 (若 n_clusters >= N 自动裁剪)
    n_clusters = min(params.get("n_clusters", 16), len(node_vectors))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(256, len(node_vectors)),
        max_iter=100,
    )
    labels = kmeans.fit_predict(node_vectors)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    cluster_members: List[List[int]] = [[] for _ in range(n_clusters)]
    for idx, cid in enumerate(labels):
        cluster_members[int(cid)].append(node_ids[idx])

    return SharedContext(
        base_index=base_index,
        node_ids=node_ids,
        node_vectors=node_vectors,
        centroids=centroids,
        cluster_members=cluster_members,
        params=params,
    )


# ------------------------------------------------------------
# 策略接口 & 实现
# ------------------------------------------------------------

class BaseAssignmentStrategy:
    name = "base"

    def prepare(self, shared: SharedContext):  # 可选
        pass

    def assign_children(
        self,
        cluster_id: int,
        centroid_vec: np.ndarray,
        shared: SharedContext,
        k_children: int,
        child_search_ef: int,
    ) -> List[int]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:  # basic metadata
        return {"strategy": self.name}


class SinglePivotStrategy(BaseAssignmentStrategy):
    """Single pivot strategy: cluster centroid is the pivot; assign k_children via HNSW global search.
    
    This implementation uses HNSW dynamic querying across the full index (like v1.py),
    rather than limiting to pre-assigned cluster members.
    """
    name = "single"

    def assign_children(self, cluster_id, centroid_vec, shared, k_children, child_search_ef):  # type: ignore[override]
        # Use HNSW to dynamically query the full index with the centroid
        # This matches the behavior in v1.py OptimizedSinglePivotSystem
        try:
            neighbors = shared.base_index.query(
                centroid_vec,
                k=k_children,
                ef=child_search_ef
            )
            # Extract node IDs from HNSW query results
            return [node_id for node_id, _ in neighbors]
        except Exception as e:
            # Fallback to cluster-member-only approach if HNSW query fails
            print(f"Warning: HNSW query failed for cluster {cluster_id}, falling back to cluster members: {e}")
            members = shared.cluster_members[cluster_id]
            if len(members) <= k_children:
                return members
            nid_to_vec_idx = {nid: i for i, nid in enumerate(shared.node_ids)}
            scored: List[Tuple[float, int]] = []
            for nid in members:
                vec = shared.node_vectors[nid_to_vec_idx[nid]]
                scored.append((l2(vec, centroid_vec), nid))
            scored.sort(key=lambda x: x[0])
            return [nid for _, nid in scored[:k_children]]


class MultiPivotStrategy(BaseAssignmentStrategy):
    """Multi-pivot diversification inside each cluster.

    Picks several pivots (either random or farthest-point selection) then scores
    each member by min distance to any pivot; take top k_children (closest to some pivot).
    (Simplified vs earlier complex logic.)
    """

    name = "multi"

    def __init__(self, num_pivots: int = 3, pivot_strategy: str = "line_perp_third"):
        self.num_pivots = max(1, num_pivots)
        self.pivot_strategy = pivot_strategy

    def metadata(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"strategy": self.name, "num_pivots": self.num_pivots, "pivot_strategy": self.pivot_strategy}

    def assign_children(self, cluster_id, centroid_vec, shared, k_children, child_search_ef):  # type: ignore[override]
        members = shared.cluster_members[cluster_id]
        if len(members) <= k_children:
            return members
        nid_to_vec_idx = {nid: i for i, nid in enumerate(shared.node_ids)}
        # Build vectors list for members
        member_vecs = {nid: shared.node_vectors[nid_to_vec_idx[nid]] for nid in members}

        # Select pivots
        pivots: List[np.ndarray] = []
        if self.pivot_strategy == "max_min_distance" and members:
            first = random.choice(members)
            pivots.append(member_vecs[first])
            while len(pivots) < min(self.num_pivots, len(members)):
                best_nid = None
                best_min = -1.0
                for nid in members:
                    v = member_vecs[nid]
                    dmin = min(l2(v, p) for p in pivots)
                    if dmin > best_min:
                        best_min = dmin
                        best_nid = nid
                if best_nid is None:
                    break
                pivots.append(member_vecs[best_nid])
        else:
            sample = random.sample(members, min(self.num_pivots, len(members)))
            pivots = [member_vecs[nid] for nid in sample]

        # Score each member by min distance to any pivot
        scored: List[Tuple[float, int]] = []
        for nid, vec in member_vecs.items():
            dmin = min(l2(vec, p) for p in pivots)
            scored.append((dmin, nid))
        scored.sort(key=lambda x: x[0])
        return [nid for _, nid in scored[:k_children]]


class HybridStrategy(BaseAssignmentStrategy):
    """Hybrid HNSW 父节点选择：优先使用指定 HNSW 层级(默认 level=2) 的节点作为父集合。

    逻辑顺序：
    1. 强制使用指定层级 (len(_graphs) > level) 且该层节点数 >=1，取该层全部节点为 parents。
    2. 若该层不存在或为空 -> 直接抛出异常，终止构建（不再做随机采样回退）。
    3. 对每个 parent 运行 HNSW 查询获取 fanout 子节点集合（排除自身）。
    4. 覆盖 shared.centroids / cluster_members 以复用统一 TwoStageIndex。
    """

    name = "hybrid"

    def __init__(
        self,
        parent_sample_size: Optional[int] = None,
        fanout: int = 64,
        ef: Optional[int] = None,
        random_seed: int = 42,
        use_level: int = 2,
    ):
        self.parent_sample_size = parent_sample_size
        self.fanout = fanout
        self.ef = ef
        self.random_seed = random_seed
        self.use_level = use_level
        self._prepared = False
        self._parent_ids: List[int] = []
        self._parent_children: List[List[int]] = []
        self._parent_source = "unknown"

    def metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "parent_sample_size": self.parent_sample_size,
            "fanout": self.fanout,
            "used_level": self.use_level,
            "parent_source": self._parent_source,
            "parent_count": len(self._parent_ids) if self._parent_ids else None,
        }

    def _parents_from_level(self, shared: SharedContext) -> Optional[List[int]]:
        base = shared.base_index
        graphs = getattr(base, "_graphs", [])
        if self.use_level < 0 or self.use_level >= len(graphs):
            return None
        layer = graphs[self.use_level]
        # layer 可迭代其键；过滤掉软删除节点
        candidates = [nid for nid in layer if (nid in base._nodes and not base._nodes[nid].is_deleted)]
        return candidates if candidates else None

    def prepare(self, shared: SharedContext):  # type: ignore[override]
        if self._prepared:
            return
        # 强制层级父节点，不存在直接抛错
        parent_ids = self._parents_from_level(shared)
        if not parent_ids:
            raise ValueError(
                f"HybridStrategy: required HNSW level {self.use_level} not available or empty; cannot fallback to sampling per request."
            )
        self._parent_source = f"level_{self.use_level}"

        base = shared.base_index
        nid_to_vec_idx = {nid: i for i, nid in enumerate(shared.node_ids)}
        parent_children: List[List[int]] = []
        ef = self.ef or max(self.fanout + 10, int(self.fanout * 1.5))
        for pid in parent_ids:
            try:
                vec = shared.node_vectors[nid_to_vec_idx[pid]]
                res = base.query(vec, k=self.fanout, ef=ef)
                kids = [nid for nid, _ in res if nid != pid]
            except Exception:
                kids = []
            parent_children.append(kids)
        shared.centroids = np.vstack([shared.node_vectors[nid_to_vec_idx[pid]] for pid in parent_ids]).astype(np.float32)
        shared.cluster_members = parent_children  # type: ignore
        self._parent_ids = parent_ids
        self._parent_children = parent_children
        self._prepared = True

    def assign_children(self, cluster_id, centroid_vec, shared, k_children, child_search_ef):  # type: ignore[override]
        members = shared.cluster_members[cluster_id]
        if len(members) <= k_children:
            return members
        nid_to_vec_idx = {nid: i for i, nid in enumerate(shared.node_ids)}
        scored = []
        for nid in members:
            vec = shared.node_vectors[nid_to_vec_idx[nid]]
            scored.append((l2(vec, centroid_vec), nid))
        scored.sort(key=lambda x: x[0])
        return [nid for _, nid in scored[:k_children]]


# ------------------------------------------------------------
# 统一两阶段索引
# ------------------------------------------------------------

class TwoStageIndex:
    def __init__(self, shared: SharedContext, strategy: BaseAssignmentStrategy, adaptive: Optional[Dict[str, Any]] = None):
        self.shared = shared
        self.strategy = strategy
        self.adaptive = adaptive or {}
        self.parent_child: Dict[str, List[int]] = {}
        # instrumentation fields (captured during _build)
        self._pre_repair_child_total: Optional[int] = None
        self._post_repair_child_total: Optional[int] = None
        self._pre_repair_coverage: Optional[float] = None
        self._post_repair_coverage: Optional[float] = None
        self.search_times: List[float] = []
        self._build()

    def _build(self) -> None:
        k_children = self.shared.params.get("k_children", 100)
        child_search_ef = self.shared.params.get("child_search_ef") or max(k_children + 10, int(k_children * 1.3))
        # strategy prepare hook (e.g. Hybrid) if present
        if hasattr(self.strategy, "prepare"):
            try:
                self.strategy.prepare(self.shared)  # type: ignore
            except Exception:
                pass
        centroids = self.shared.centroids
        for cid, centroid_vec in enumerate(centroids):
            children = self.strategy.assign_children(cid, centroid_vec, self.shared, k_children, child_search_ef)
            self.parent_child[f"centroid_{cid}"] = children
        # pre-repair instrumentation
        self._pre_repair_child_total = sum(len(v) for v in self.parent_child.values())
        pre_stats = compute_assignment_stats(self.parent_child, self.shared.base_index)
        self._pre_repair_coverage = pre_stats.get("coverage_fraction")
        # optional repair
        if self.adaptive.get("repair_min_assignments"):
            self._repair(self.adaptive["repair_min_assignments"])
        # post-repair instrumentation
        self._post_repair_child_total = sum(len(v) for v in self.parent_child.values())
        post_stats = compute_assignment_stats(self.parent_child, self.shared.base_index)
        self._post_repair_coverage = post_stats.get("coverage_fraction")

    def _repair(self, min_assign: int) -> None:
        counts: Dict[int, int] = {}
        for children in self.parent_child.values():
            for c in children:
                counts[c] = counts.get(c, 0) + 1
        all_nodes = set(self.shared.node_ids)
        need = {nid for nid in all_nodes if counts.get(nid, 0) < min_assign}
        if not need:
            return
        centroid_matrix = self.shared.centroids
        for nid in need:
            vec = self.shared.node_vectors[self.shared.node_ids.index(nid)]
            dists = np.linalg.norm(centroid_matrix - vec, axis=1)
            best_cid = int(np.argmin(dists))
            key = f"centroid_{best_cid}"
            if nid not in self.parent_child[key]:
                self.parent_child[key].append(nid)

    def search(self, query: np.ndarray, k: int = 10, n_probe: int = 5) -> List[Tuple[int, float]]:
        t0 = time.time()
        centroid_matrix = self.shared.centroids
        dists = np.linalg.norm(centroid_matrix - query, axis=1)
        probe_idx = np.argsort(dists)[: min(n_probe, len(centroid_matrix))]
        candidate_ids: List[int] = []
        for idx in probe_idx:
            candidate_ids.extend(self.parent_child.get(f"centroid_{idx}", []))
        candidate_ids = list(dict.fromkeys(candidate_ids))  # deduplicate
        if not candidate_ids:
            return []
        nid_to_vec_idx = {nid: i for i, nid in enumerate(self.shared.node_ids)}
        vecs = np.vstack([self.shared.node_vectors[nid_to_vec_idx[nid]] for nid in candidate_ids])
        cdists = np.linalg.norm(vecs - query, axis=1)
        order = np.argsort(cdists)[:k]
        self.search_times.append((time.time() - t0) * 1000.0)
        return [(candidate_ids[i], float(cdists[i])) for i in order]

    def get_stats(self) -> Dict[str, Any]:
        base_stats = compute_assignment_stats(self.parent_child, self.shared.base_index)
        if self.search_times:
            base_stats.update(
                {
                    "avg_search_time_ms": float(np.mean(self.search_times)),
                    "std_search_time_ms": float(np.std(self.search_times)),
                }
            )
        base_stats.update(self.strategy.metadata())
        base_stats.update(
            {
                "n_clusters": int(self.shared.centroids.shape[0]),
                "k_children": int(self.shared.params.get("k_children", 100)),
                "pre_repair_child_total": self._pre_repair_child_total,
                "post_repair_child_total": self._post_repair_child_total,
                "pre_repair_coverage": self._pre_repair_coverage,
                "post_repair_coverage": self._post_repair_coverage,
            }
        )
        return base_stats


# ------------------------------------------------------------
# 评估器
# ------------------------------------------------------------

class Evaluator:
    def __init__(self, dataset: np.ndarray, queries: np.ndarray, query_ids: List[int]):
        self.dataset = dataset
        self.queries = queries
        self.query_ids = query_ids
        self._gt_cache: Dict[Tuple[int, bool], Dict[int, List[Tuple[int, float]]]] = {}

    def ground_truth(self, k: int, exclude_self: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        key = (k, exclude_self)
        if key in self._gt_cache:
            return self._gt_cache[key]
        res: Dict[int, List[Tuple[int, float]]] = {}
        for qvec, qid in zip(self.queries, self.query_ids):
            dists: List[Tuple[float, int]] = []
            for idx, dvec in enumerate(self.dataset):
                if exclude_self and idx == qid:
                    continue
                dists.append((l2(qvec, dvec), idx))
            dists.sort(key=lambda x: x[0])
            res[qid] = dists[:k]
        self._gt_cache[key] = res
        return res

    def evaluate(self, system: TwoStageIndex, k: int, n_probe: int, gt: Dict[int, List[Tuple[int, float]]]) -> Dict[str, Any]:
        total = 0
        correct = 0
        recalls: List[float] = []
        times: List[float] = []
        for qvec, qid in zip(self.queries, self.query_ids):
            true_ids = {nid for _, nid in gt[qid]}
            t0 = time.time()
            results = system.search(qvec, k=k, n_probe=n_probe)
            times.append((time.time() - t0) * 1000.0)
            found = {nid for nid, _ in results}
            inter = len(true_ids & found)
            correct += inter
            total += k
            recalls.append(inter / k if k else 0.0)
        stats = system.get_stats()
        return {
            "k": k,
            "n_probe": n_probe,
            "recall_at_k": correct / total if total else 0.0,
            "avg_individual_recall": float(np.mean(recalls)),
            "std_individual_recall": float(np.std(recalls)),
            "avg_query_time_ms": float(np.mean(times)),
            "std_query_time_ms": float(np.std(times)),
            "system_stats": stats,
        }


# ------------------------------------------------------------
# 命令行示例
# ------------------------------------------------------------

def main():  # pragma: no cover
    p = argparse.ArgumentParser("Unified KMeans + (Multi)Pivot + HNSW evaluator")
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--query-size", type=int, default=20)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--n-clusters", type=int, default=16)
    p.add_argument("--k-children", type=int, default=200)
    p.add_argument("--child-search-ef", type=int, default=None)
    p.add_argument("--strategy", type=str, default="all", choices=["single", "multi", "hybrid", "all"])
    p.add_argument("--num-pivots", type=int, default=3)
    p.add_argument("--pivot-strategy", type=str, default="line_perp_third", choices=["line_perp_third", "max_min_distance"])
    p.add_argument("--repair-min", type=int, default=None)
    p.add_argument("--hybrid-parent-sample", type=int, default=None, help="Hybrid parent sample size (default sqrt(N))")
    p.add_argument("--hybrid-fanout", type=int, default=64, help="Hybrid each parent queries this many children")
    p.add_argument("--hybrid-level", type=int, default=2, help="Hybrid: use this HNSW level as parent set if available")
    p.add_argument("--baseline-ef", type=int, default=200, help="ef parameter for baseline HNSW recall evaluation")
    p.add_argument("--k-list", type=str, default="10", help="Comma separated k values for recall evaluation e.g. 10,20")
    p.add_argument("--n-probe-list", type=str, default="5,10", help="Comma separated n_probe values e.g. 5,10,20")
    p.add_argument("--out", type=str, default="optimized_multi_pivot_results.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Building synthetic dataset: N={args.dataset_size}, Q={args.query_size}, dim={args.dimension}")
    dataset = np.random.randn(args.dataset_size, args.dimension).astype(np.float32)
    queries = np.random.randn(args.query_size, args.dimension).astype(np.float32)
    query_ids = list(range(len(queries)))

    # 构建 HNSW 基础索引
    print("Building base HNSW index...")
    hnsw = HNSW(distance_func=lambda a, b: np.linalg.norm(a - b), m=16, ef_construction=200)
    for i, vec in enumerate(dataset):
        hnsw.insert(i, vec)
        if (i + 1) % 500 == 0:
            print(f"  inserted {i + 1}/{len(dataset)}")
    print(f"Base index size = {len(hnsw)}")

    params = {
        "n_clusters": args.n_clusters,
        "k_children": args.k_children,
        "child_search_ef": args.child_search_ef,
    }
    shared = build_shared(hnsw, params)
    evaluator = Evaluator(dataset, queries, query_ids)
    k_list = [int(x) for x in args.k_list.split(',') if x.strip()]
    n_probe_list = [int(x) for x in args.n_probe_list.split(',') if x.strip()]
    # 预计算所有 k 的 ground truth
    gt_cache: Dict[int, Dict[int, List[Tuple[int, float]]]] = {}
    for k_val in k_list:
        gt_cache[k_val] = evaluator.ground_truth(k=k_val, exclude_self=False)

    strategies: List[BaseAssignmentStrategy] = []
    if args.strategy in ("single", "all"):
        strategies.append(SinglePivotStrategy())
    if args.strategy in ("multi", "all"):
        strategies.append(MultiPivotStrategy(num_pivots=args.num_pivots, pivot_strategy=args.pivot_strategy))
    if args.strategy in ("hybrid", "all"):
        strategies.append(HybridStrategy(parent_sample_size=args.hybrid_parent_sample, fanout=args.hybrid_fanout, use_level=args.hybrid_level))

    methods: Dict[str, Any] = {}
    for strat in strategies:
        print(f"Building system with strategy={strat.metadata()}")
        sys = TwoStageIndex(shared, strat, adaptive={"repair_min_assignments": args.repair_min})
        strat_evals = []
        for k_val in k_list:
            for n_probe in n_probe_list:
                res = evaluator.evaluate(sys, k=k_val, n_probe=n_probe, gt=gt_cache[k_val])
                strat_evals.append(res)
        # 汇总：最佳 recall, 平均查询时间
        best = max(strat_evals, key=lambda r: r['recall_at_k']) if strat_evals else None
        methods[strat.name] = {
            "strategy": strat.metadata(),
            "system_stats": sys.get_stats(),
            "evaluations": strat_evals,
            "best_recall": best['recall_at_k'] if best else None,
        }
        if best:
            print(f"  best recall={best['recall_at_k']:.4f} (k={best['k']}, n_probe={best['n_probe']})")

    # Baseline HNSW recall
    def eval_baseline(hnsw_index: HNSW, k: int, ef: int, queries: np.ndarray, qids: List[int], gt: Dict[int, List[Tuple[int, float]]]):
        times = []
        total = 0
        correct = 0
        per = []
        for qvec, qid in zip(queries, qids):
            t0 = time.time()
            try:
                res = hnsw_index.query(qvec, k=k, ef=ef)
            except Exception:
                res = []
            times.append((time.time() - t0) * 1000.0)
            found = {nid for nid, _ in res}
            true_ids = {nid for _, nid in gt[qid]}
            inter = len(found & true_ids)
            correct += inter
            total += k
            per.append(inter / k if k else 0.0)
        return {
            "phase": "baseline_hnsw",
            "k": k,
            "n_probe": None,  # 对齐其他策略的 schema，baseline 无 n_probe 概念
            "ef": ef,
            "recall_at_k": (correct / total) if total else 0.0,
            "avg_individual_recall": float(np.mean(per)),
            "std_individual_recall": float(np.std(per)),
            "avg_query_time_ms": float(np.mean(times)),
            "std_query_time_ms": float(np.std(times)),
            "coverage_fraction": 1.0,
        }

    # Baseline: evaluate for each k
    baseline_evals = []
    for k_val in k_list:
        baseline_evals.append(eval_baseline(hnsw, k=k_val, ef=args.baseline_ef, queries=queries, qids=query_ids, gt=gt_cache[k_val]))
    best_base = max(baseline_evals, key=lambda r: r['recall_at_k']) if baseline_evals else None
    methods["baseline_hnsw"] = {
        "strategy": {"strategy": "baseline_hnsw", "ef": args.baseline_ef},
        "system_stats": {"coverage_fraction": 1.0, "total_nodes": len(hnsw)},
        "evaluations": baseline_evals,
        "best_recall": best_base['recall_at_k'] if best_base else None,
    }

    output_payload = {
        "dataset": {"size": len(dataset), "queries": len(queries), "dimension": args.dimension},
        "params": {"n_clusters": args.n_clusters, "k_children": args.k_children},
        "k_list": k_list,
        "n_probe_list": n_probe_list,
        "methods": methods,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 统计：重复率 vs recall / 覆盖率 vs recall (取每 evaluation 记录 after system stats duplication_rate & coverage_fraction)
    dup_points: List[Tuple[float, float]] = []
    cover_points: List[Tuple[float, float]] = []
    for m_name, mdata in methods.items():
        if m_name == 'baseline_hnsw':
            # baseline 没有重复分配语义
            for ev in mdata['evaluations']:
                cover_points.append((1.0, ev['recall_at_k']))
            continue
        stats = mdata.get('system_stats', {})
        dup_rate = stats.get('duplication_rate')
        cover = stats.get('coverage_fraction')
        if dup_rate is not None:
            # 用最佳 recall 代表该方法点
            dup_points.append((dup_rate, mdata.get('best_recall', 0.0)))
        if cover is not None:
            cover_points.append((cover, mdata.get('best_recall', 0.0)))

    def pearson(pairs: List[Tuple[float, float]]):
        if len(pairs) < 2:
            return None
        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)
        if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])

    output_payload['correlation'] = {
        'duplication_vs_recall_pearson': pearson(dup_points),
        'coverage_vs_recall_pearson': pearson(cover_points),
        'duplication_samples': len(dup_points),
        'coverage_samples': len(cover_points),
    }

    # CSV 导出: evaluations_flat.csv & methods_summary.csv
    import csv, os
    base_out = os.path.splitext(args.out)[0]
    eval_csv = base_out + '_evaluations.csv'
    summary_csv = base_out + '_methods_summary.csv'

    # 展开 evaluations
    eval_headers = [
        'method','k','n_probe','recall_at_k','avg_individual_recall','std_individual_recall',
        'avg_query_time_ms','std_query_time_ms','duplication_rate','coverage_fraction'
    ]
    with open(eval_csv, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(eval_headers)
        for m_name, mdata in methods.items():
            dup_rate = mdata.get('system_stats', {}).get('duplication_rate')
            cover = mdata.get('system_stats', {}).get('coverage_fraction')
            for ev in mdata['evaluations']:
                writer.writerow([
                    m_name,
                    ev['k'],
                    ev.get('n_probe',''),
                    f"{ev['recall_at_k']:.6f}",
                    f"{ev['avg_individual_recall']:.6f}",
                    f"{ev['std_individual_recall']:.6f}",
                    f"{ev['avg_query_time_ms']:.6f}",
                    f"{ev['std_query_time_ms']:.6f}",
                    f"{dup_rate:.6f}" if dup_rate is not None else '',
                    f"{cover:.6f}" if cover is not None else '',
                ])

    # 方法摘要
    summary_headers = ['method','best_recall','duplication_rate','coverage_fraction','total_assignments','unique_assigned_nodes']
    with open(summary_csv, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(summary_headers)
        for m_name, mdata in methods.items():
            stats = mdata.get('system_stats', {})
            writer.writerow([
                m_name,
                f"{mdata.get('best_recall', 0.0):.6f}",
                f"{stats.get('duplication_rate', 0.0):.6f}" if 'duplication_rate' in stats else '',
                f"{stats.get('coverage_fraction', 0.0):.6f}" if 'coverage_fraction' in stats else '',
                stats.get('total_assignments',''),
                stats.get('unique_assigned_nodes',''),
            ])

    output_payload['csv_outputs'] = {
        'evaluations': eval_csv,
        'methods_summary': summary_csv,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved results -> {args.out}")
    print(f"CSV evaluations -> {eval_csv}")
    print(f"CSV summary -> {summary_csv}")

    # ------------------------------------------------------------
    # Inline summary print (requested metrics)
    # ------------------------------------------------------------
    if 'hybrid' in methods and 'single' in methods:
        h_stats = methods['hybrid']['system_stats']
        s_stats = methods['single']['system_stats']
        print("\n==== Summary (Hybrid vs Single) ====")
        print(f"Hybrid parent_count={h_stats.get('parent_count')} fanout={h_stats.get('fanout')} level_used={h_stats.get('used_level')}")
        print(f"Hybrid pre/post child_total: {h_stats.get('pre_repair_child_total')} -> {h_stats.get('post_repair_child_total')} (Δ={h_stats.get('post_repair_child_total') - h_stats.get('pre_repair_child_total',0) if h_stats.get('pre_repair_child_total') is not None else 'NA'})")
        if h_stats.get('pre_repair_child_total'):
            delta_pct = (h_stats['post_repair_child_total'] - h_stats['pre_repair_child_total']) / h_stats['pre_repair_child_total'] * 100.0
            print(f"  (+{delta_pct:.2f}% assignments after repair)")
        print(f"Hybrid coverage: {h_stats.get('pre_repair_coverage')} -> {h_stats.get('post_repair_coverage')}  duplication_rate={h_stats.get('duplication_rate'):.4f}")
        print(f"Single pre/post child_total: {s_stats.get('pre_repair_child_total')} -> {s_stats.get('post_repair_child_total')}")
        print(f"Single coverage: {s_stats.get('pre_repair_coverage')} -> {s_stats.get('post_repair_coverage')}  duplication_rate={s_stats.get('duplication_rate'):.4f}")
        # Recall per n_probe (k first element of k_list)
        k_val = k_list[0]
        def collect_recall(m_name: str):
            ret = {}
            for ev in methods[m_name]['evaluations']:
                if ev['k'] == k_val:
                    ret[ev['n_probe']] = ev['recall_at_k']
            return ret
        h_recall = collect_recall('hybrid')
        s_recall = collect_recall('single')
        probes = sorted(set(h_recall.keys()) | set(s_recall.keys()))
        print("n_probe  single  hybrid  delta  hybrid_parent_cover  single_parent_cover")
        for p in probes:
            sr = s_recall.get(p)
            hr = h_recall.get(p)
            delta = (hr - sr) if (hr is not None and sr is not None) else None
            # parent counts
            h_pc = h_stats.get('parent_count') or 1
            s_pc = s_stats.get('n_clusters') or 1
            h_cover = f"{p/h_pc:.3f}" if h_pc else 'NA'
            s_cover = f"{p/s_pc:.3f}" if s_pc else 'NA'
            print(f"{p:>6}  {sr:.3f}  {hr:.3f}  {delta:+.3f}  {h_cover}  {s_cover}")
        print("===================================\n")


if __name__ == "__main__":  # pragma: no cover
    main()
