from __future__ import annotations
from collections import OrderedDict
import heapq
from typing import (
    Hashable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np


class _Layer(object):
    """HNSW索引中的图层级。这是一个类似字典的对象，
    将键映射到邻居字典。

    HNSW使用多层图结构，其中：
    - 较高层级的节点数量较少（指数级递减）
    - 每一层都是一个图，节点连接到它们的最近邻
    - 底层（第0层）包含所有节点
    - 较高层级用于搜索期间的快速导航

    Args:
        key (Hashable): 要插入到图中的第一个键。
    """

    def __init__(self, key: Hashable) -> None:
        # 初始化此层的图结构
        # self._graph[key] 包含一个 {neighbor_key: distance} 字典，
        # 其中 neighbor_key 是 key 的邻居，distance 是它们之间的距离
        # 这表示此层中每个节点的出边
        self._graph: Dict[Hashable, Dict[Hashable, float]] = {key: {}}

    def __contains__(self, key: Hashable) -> bool:
        """检查键是否存在于此层中。"""
        return key in self._graph

    def __getitem__(self, key: Hashable) -> Dict[Hashable, float]:
        """获取键的邻居字典：{neighbor_key: distance}。"""
        return self._graph[key]

    def __setitem__(self, key: Hashable, value: Dict[Hashable, float]) -> None:
        """设置键的邻居字典：{neighbor_key: distance}。"""
        self._graph[key] = value

    def __delitem__(self, key: Hashable) -> None:
        """从此层中移除键及其所有出边。"""
        del self._graph[key]

    def __eq__(self, __value: object) -> bool:
        """检查两个层是否具有相同的图结构。"""
        if not isinstance(__value, _Layer):
            return False
        return self._graph == __value._graph

    def __len__(self) -> int:
        """返回此层中的节点数量。"""
        return len(self._graph)

    def __iter__(self) -> Iterable[Hashable]:
        """迭代此层中的所有键。"""
        return iter(self._graph)

    def copy(self) -> _Layer:
        """创建包含所有连接的层的深拷贝。"""
        new_layer = _Layer(None)
        # 深拷贝图结构以避免共享引用
        new_layer._graph = {k: dict(v) for k, v in self._graph.items()}
        return new_layer

    def get_reverse_edges(self, key: Hashable) -> Set[Hashable]:
        """查找所有指向给定键的出边的节点。
        
        这在删除期间用于查找当节点被移除时需要更新连接的节点。
        
        Args:
            key: 要查找反向边的键
            
        Returns:
            具有指向给定键的出边的键集合
        """
        reverse_edges = set()
        # 遍历此层中的所有节点
        for neighbor, neighbors in self._graph.items():
            # 如果此邻居有指向我们目标键的边，将其添加到反向边中
            if key in neighbors:
                reverse_edges.add(neighbor)
        return reverse_edges


class _LayerWithReversedEdges(_Layer):
    """HNSW索引中同时维护反向边的图层级。

    这是_Layer的优化版本，同时维护正向和反向边映射。这使得硬删除操作更快，
    因为我们可以快速找到所有指向被删除节点的节点，而无需扫描整个图。
    但是，它使用更多内存并略微减慢插入速度。

    权衡：
    - 内存：由于反向边存储，内存使用量增加约2倍
    - 插入：由于维护反向边，插入速度略微减慢
    - 删除：硬删除操作快得多

    Args:
        key (Hashable): 要插入到图中的第一个键。
    """

    def __init__(self, key: Hashable) -> None:
        # 正向边：self._graph[key] 包含 {neighbor_key: distance} 字典
        # 这表示每个节点的出边
        self._graph: Dict[Hashable, Dict[Hashable, float]] = {key: {}}
        
        # 反向边：self._reverse_edges[key] 包含具有指向此键的出边的节点集合
        # 这允许快速查找入边
        self._reverse_edges: Dict[Hashable, Set] = {}

    def __setitem__(self, key: Hashable, value: Dict[Hashable, float]) -> None:
        """为键设置邻居并更新反向边映射。
        
        此方法维护正向和反向边的一致性。
        当我们更新节点的邻居时，我们需要：
        1. 移除旧的反向边（不再指向此键的节点）
        2. 添加新的反向边（现在指向此键的节点）
        """
        # 在更新前获取旧的邻居
        old_neighbors = self._graph.get(key, {})
        
        # 更新正向边
        self._graph[key] = value
        
        # 移除不再连接的旧邻居的反向边
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)
            
        # 为新邻居添加反向边
        for neighbor in value:
            self._reverse_edges.setdefault(neighbor, set()).add(key)
            
        # 确保键本身有一个反向边条目（即使为空）
        if key not in self._reverse_edges:
            self._reverse_edges[key] = set()

    def __delitem__(self, key: Hashable) -> None:
        """移除键并清理所有相关的边映射。"""
        # 在删除前获取旧邻居
        old_neighbors = self._graph.get(key, {})
        
        # 从正向图中移除
        del self._graph[key]
        
        # 清理反向边 - 从所有指向它的节点中移除此键
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)

    def __eq__(self, __value: object) -> bool:
        """检查两个带反向边的层是否具有相同的结构。"""
        if not isinstance(__value, _LayerWithReversedEdges):
            return False
        return (
            self._graph == __value._graph
            and self._reverse_edges == __value._reverse_edges
        )

    def __len__(self) -> int:
        """返回此层中的节点数量。"""
        return len(self._graph)

    def __iter__(self) -> Iterable[Hashable]:
        """迭代此层中的所有键。"""
        return iter(self._graph)

    def copy(self) -> _LayerWithReversedEdges:
        """创建包含所有正向和反向边的层的深拷贝。"""
        new_layer = _LayerWithReversedEdges(None)
        # 深拷贝正向和反向边结构
        new_layer._graph = {k: dict(v) for k, v in self._graph.items()}
        new_layer._reverse_edges = {k: set(v) for k, v in self._reverse_edges.items()}
        return new_layer

    def get_reverse_edges(self, key: Hashable) -> Set[Hashable]:
        """获取所有具有指向给定键的出边的节点。
        
        这比基类实现快得多，因为我们显式维护反向边，
        而不是扫描整个图。
        
        Args:
            key: 要查找反向边的键
            
        Returns:
            具有指向给定键的出边的键集合
        """
        return self._reverse_edges[key]


class _Node(object):
    """HNSW图中表示单个数据点的节点。
    
    每个节点包含：
    - key: 节点的唯一标识符
    - point: 实际的数据向量（numpy数组）
    - is_deleted: 软删除标志（节点存在但被标记为已删除）
    """

    def __init__(self, key: Hashable, point: np.ndarray, is_deleted=False) -> None:
        """使用键、数据点和删除状态初始化节点。
        
        Args:
            key: 此节点的唯一标识符
            point: 数据向量（numpy数组）
            is_deleted: 此节点是否被软删除（默认：False）
        """
        self.key = key
        self.point = point
        self.is_deleted = is_deleted

    def __eq__(self, __value: object) -> bool:
        """检查两个节点是否相等（相同的键、点和删除状态）。"""
        if not isinstance(__value, _Node):
            return False
        return (
            self.key == __value.key
            and np.array_equal(self.point, __value.point)
            and self.is_deleted == __value.is_deleted
        )

    def __hash__(self) -> int:
        """仅基于键进行哈希（用于集合/字典）。"""
        return hash(self.key)

    def __repr__(self) -> str:
        """节点的字符串表示，用于调试。"""
        return (
            f"_Node(key={self.key}, point={self.point}, is_deleted={self.is_deleted})"
        )

    def copy(self) -> _Node:
        """创建具有相同属性的此节点的副本。"""
        return _Node(self.key, self.point, self.is_deleted)


class HNSW(MutableMapping):
    """Hierarchical Navigable Small World (HNSW) graph index for approximate
    nearest neighbor search. This implementation is based on the paper
    "Efficient and robust approximate nearest neighbor search using Hierarchical
    Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin (2016),
    `<https://arxiv.org/abs/1603.09320>`_.

    Args:
        distance_func: A function that takes two vectors and returns a float
            representing the distance between them.
        m (int): The number of neighbors to keep for each node.
        ef_construction (int): The number of neighbors to consider during
            construction.
        m0 (Optional[int]): The number of neighbors to keep for each node at
            the 0th level. If None, defaults to 2 * m.
        seed (Optional[int]): The random seed to use for the random number
            generator.
        reverse_edges (bool): Whether to maintain reverse edges in the graph.
            This speeds up hard remove (:meth:`remove`) but increases memory
            usage and slows down :meth:`insert`.

    Examples:

        Create an HNSW index with Euclidean distance and insert 1000 random
        vectors of dimension 10.

        .. code-block:: python

            from datasketch.hnsw import HNSW
            import numpy as np

            data = np.random.random_sample((1000, 10))
            index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
            for i, d in enumerate(data):
                index.insert(i, d)

            # Query the index for the 10 nearest neighbors of the first vector.
            index.query(data[0], k=10)

        Create an HNSW index with Jaccard distance and insert 1000 random
        sets of 10 elements each.

        .. code-block:: python

            from datasketch.hnsw import HNSW
            import numpy as np

            # Each set is represented as a 10-element vector of random integers
            # between 0 and 100.
            # Deduplication is handled by the distance function.
            data = np.random.randint(0, 100, size=(1000, 10))
            jaccard_distance = lambda x, y: (
                1.0 - float(len(np.intersect1d(x, y, assume_unique=False)))
                / float(len(np.union1d(x, y)))
            )
            index = HNSW(distance_func=jaccard_distance)
            for i, d in enumerate(data):
                index[i] = d

            # Query the index for the 10 nearest neighbors of the first set.
            index.query(data[0], k=10)

    """

    def __init__(
        self,
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        m: int = 16,
        ef_construction: int = 200,
        m0: Optional[int] = None,
        seed: Optional[int] = None,
        reversed_edges: bool = False,
    ) -> None:
        """使用给定参数初始化HNSW索引。
        
        Args:
            distance_func: 计算两个向量之间距离的函数
            m: 每个节点的最大邻居数（默认：16）
            ef_construction: 构建期间考虑的候选数（默认：200）
            m0: 第0层的最大邻居数（默认：2*m）
            seed: 用于可重现结果的随机种子
            reversed_edges: 是否维护反向边以加快删除速度
        """
        # 索引中所有节点的存储（保持插入顺序）
        self._nodes: OrderedDict[Hashable, _Node] = OrderedDict()
        
        # 计算向量之间相似性的距离函数
        self._distance_func = distance_func
        
        # HNSW算法参数
        self._m = m  # 每个节点的最大邻居数（除第0层外）
        self._ef_construction = ef_construction  # 构建搜索宽度
        self._m0 = 2 * m if m0 is None else m0  # 第0层的最大邻居数
        
        # 层级分配乘数：level = int(-log(random()) * level_mult)
        # 这控制节点层级的概率分布
        self._level_mult = 1 / np.log(m)
        
        # 图层级列表（较高索引 = 较高层级）
        # 第0层包含所有节点，较高层级具有指数级更少的节点
        self._graphs: List[_Layer] = []
        
        # 搜索的入口点（最高层级的节点）
        self._entry_point = None
        
        # 用于层级分配的随机数生成器
        self._random = np.random.RandomState(seed)
        
        # 根据是否要反向边选择层级类
        self._layer_class = _LayerWithReversedEdges if reversed_edges else _Layer

    def __len__(self) -> int:
        """返回索引中活动（未删除）点的数量。
        
        这只计算未被软删除的节点，提供索引的有效大小
        用于查询和操作。
        """
        return sum(not node.is_deleted for node in self._nodes.values())

    def __contains__(self, key: Hashable) -> bool:
        """检查键是否存在于索引中且未被软删除。
        
        仅当键存在且未被标记为删除时才返回True。
        软删除的节点不被认为"在"索引中。
        """
        return key in self._nodes and not self._nodes[key].is_deleted

    def __getitem__(self, key: Hashable) -> np.ndarray:
        """获取与键关联的数据点。
        
        如果键不存在或被软删除，则引发KeyError。
        这提供了对存储向量的类似字典的访问。
        """
        if key not in self:
            raise KeyError(key)
        return self._nodes[key].point

    def __setitem__(self, key: Hashable, value: np.ndarray) -> None:
        """使用字典语法设置/更新数据点。
        
        这等同于调用insert(key, value)，允许
        HNSW像字典一样使用：index[key] = vector
        """
        self.insert(key, value)

    def __delitem__(self, key: Hashable) -> None:
        """使用字典语法软删除点。
        
        这执行软删除（标记为已删除但保留在图中）。
        等同于调用remove(key)而不使用hard=True。
        """
        self.remove(key)

    def __iter__(self) -> Iterator[Hashable]:
        """返回索引中未被软删除的键的迭代器。"""
        return (key for key in self._nodes if not self._nodes[key].is_deleted)

    def reversed(self) -> Iterator[Hashable]:
        """返回索引中未被软删除的键的反向迭代器。"""
        return (key for key in reversed(self._nodes) if not self._nodes[key].is_deleted)

    def __eq__(self, __value: object) -> bool:
        """仅当索引参数、随机状态、键、点和索引结构都相等时返回True，
        包括已删除的点。"""
        if not isinstance(__value, HNSW):
            return False
        # 检查索引参数是否相等。
        if (
            self._distance_func != __value._distance_func
            or self._m != __value._m
            or self._ef_construction != __value._ef_construction
            or self._m0 != __value._m0
            or self._level_mult != __value._level_mult
            or self._entry_point != __value._entry_point
        ):
            return False
        # 检查随机状态是否相等。
        rand_state_1 = self._random.get_state()
        rand_state_2 = __value._random.get_state()
        for i in range(len(rand_state_1)):
            if isinstance(rand_state_1[i], np.ndarray):
                if not np.array_equal(rand_state_1[i], rand_state_2[i]):
                    return False
            else:
                if rand_state_1[i] != rand_state_2[i]:
                    return False
        # 检查键和点是否相等。
        # 注意已删除的点也会被比较。
        return (
            all(key in self._nodes for key in __value._nodes)
            and all(key in __value._nodes for key in self._nodes)
            and all(self._nodes[key] == __value._nodes[key] for key in self._nodes)
            and self._graphs == __value._graphs
        )

    def get(
        self, key: Hashable, default: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, None]:
        """返回索引中键对应的点，否则返回默认值。如果未提供默认值且键不在索引中
        或被软删除，返回None。"""
        if key not in self:
            return default
        return self._nodes[key].point

    def items(self) -> Iterator[Tuple[Hashable, np.ndarray]]:
        """返回未被软删除的索引点的迭代器，作为(key, point)对。"""
        return (
            (key, node.point)
            for key, node in self._nodes.items()
            if not node.is_deleted
        )

    def keys(self) -> Iterator[Hashable]:
        """返回未被软删除的索引点的键的迭代器。"""
        return (key for key in self._nodes if not self._nodes[key].is_deleted)

    def values(self) -> Iterator[np.ndarray]:
        """返回未被软删除的索引点的迭代器。"""
        return (node.point for node in self._nodes.values() if not node.is_deleted)

    def pop(
        self, key: Hashable, default: Optional[np.ndarray] = None, hard: bool = False
    ) -> np.ndarray:
        """如果键在索引中，移除它并返回其关联的点，否则返回默认值。
        如果未提供默认值且键不在索引中或被软删除，引发KeyError。
        """
        if key not in self:
            if default is None:
                raise KeyError(key)
            return default
        point = self._nodes[key].point
        self.remove(key, hard=hard)
        return point

    def popitem(
        self, last: bool = True, hard: bool = False
    ) -> Tuple[Hashable, np.ndarray]:
        """从索引中移除并返回一个(key, point)对。如果`last`为true，
        按LIFO顺序返回对，如果为false则按FIFO顺序返回。
        如果索引为空或所有点都被软删除，引发KeyError。

        Note:
            在Python 3.7之前的版本中，索引中项目的顺序不保证。
            此方法将移除并返回任意的(key, point)对。
        """
        if not self._nodes:
            raise KeyError("popitem(): index is empty")
        if last:
            key = next(
                (
                    key
                    for key in reversed(self._nodes)
                    if not self._nodes[key].is_deleted
                ),
                None,
            )
        else:
            key = next(
                (key for key in self._nodes if not self._nodes[key].is_deleted), None
            )
        if key is None:
            raise KeyError("popitem(): index is empty")
        point = self._nodes[key].point
        self.remove(key, hard=hard)
        return key, point

    def clear(self) -> None:
        """清空索引中的所有数据点。这不会重置随机数生成器。"""
        self._nodes = {}
        self._graphs = []
        self._entry_point = None

    def copy(self) -> HNSW:
        """创建索引的副本。副本将具有与原始索引相同的参数和相同的键和点，
        但不会与原始索引共享任何索引数据结构（即图）。
        新索引的随机状态将从原始索引的副本开始。"""
        new_index = HNSW(
            self._distance_func,
            m=self._m,
            ef_construction=self._ef_construction,
            m0=self._m0,
        )
        new_index._nodes = OrderedDict(
            (key, node.copy()) for key, node in self._nodes.items()
        )
        new_index._graphs = [layer.copy() for layer in self._graphs]
        new_index._entry_point = self._entry_point
        new_index._random.set_state(self._random.get_state())
        return new_index

    def update(self, other: Union[Mapping, HNSW]) -> None:
        """使用来自其他Mapping或HNSW对象的点更新索引，覆盖现有键。

        Args:
            other (Union[Mapping, HNSW]): 另一个Mapping或HNSW对象。

        Examples:

            使用点字典创建HNSW索引。

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # 批量插入1000个点。
                index.update({i: d for i, d in enumerate(data)})

            使用另一个HNSW索引创建HNSW索引。

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index1 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index2 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # 批量插入1000个点。
                index1.update({i: d for i, d in enumerate(data)})

                # 使用index1中的点更新index2。
                index2.update(index1)

        """
        for key, point in other.items():
            self.insert(key, point)

    def setdefault(self, key: Hashable, default: np.ndarray) -> np.ndarray:
        """如果键在索引中且未被软删除，返回其关联的点。如果没有，
        插入键并设置默认值，然后返回默认值。默认值不能为None。"""
        if default is None:
            raise ValueError("Default value cannot be None.")
        if key not in self._nodes or self._nodes[key].is_deleted:
            self.insert(key, default)
        return self._nodes[key]

    def insert(
        self,
        key: Hashable,
        new_point: np.ndarray,
        ef: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """向HNSW索引添加新点。

        这是核心插入算法，它：
        1. 确定新节点的层级（如果未指定）
        2. 搜索每层的最佳入口点
        3. 将新节点连接到其最近邻
        4. 更新现有邻居以包含新节点
        5. 维护分层结构

        Args:
            key (Hashable): 新点的键。如果键已存在于索引中，
                点将被更新，索引将被修复。
            new_point (np.ndarray): 要添加到索引的新点。
            ef (Optional[int]): 插入期间考虑的邻居数。
                如果为None，使用构建ef。
            level (Optional[int]): 插入新点的层级。
                如果为None，将使用HNSW算法自动选择层级。

        """
        # 如果未指定，使用构建ef
        if ef is None:
            ef = self._ef_construction
            
        # 处理对现有节点的更新
        if key in self._nodes:
            if self._nodes[key].is_deleted:
                # 重新激活软删除的节点
                self._nodes[key].is_deleted = False
            # 更新现有节点并修复连接
            self._update(key, new_point, ef)
            return
            
        # 使用HNSW算法确定此节点的层级
        # 较高层级具有指数级更少的节点
        if level is None:
            # 使用几何分布：P(level = l) = (1/m)^l * (1 - 1/m)
            level = int(-np.log(self._random.random_sample()) * self._level_mult)
            
        # 创建新节点
        self._nodes[key] = _Node(key, new_point)
        
        # 如果这不是第一个节点，我们需要将其连接到图中
        if self._entry_point is not None:
            # 从入口点开始（最高层级节点）
            dist = self._distance_func(new_point, self._nodes[self._entry_point].point)
            point = self._entry_point
            
            # 阶段1：使用贪婪搜索从较高层级向下导航
            # 对于插入层级之上的层级，我们只需要找到最近的节点
            # 用作下一层向下的入口点
            for layer in reversed(self._graphs[level + 1 :]):
                point, dist = self._search_ef1(
                    new_point, point, dist, layer, allow_soft_deleted=True
                )
                
            # 阶段2：从我们的层级向下到第0层的每一层插入
            # 从较高层级找到的入口点开始
            entry_points = [(-dist, point)]
            
            for layer in reversed(self._graphs[: level + 1]):
                # 确定此层的最大邻居数
                level_m = self._m if layer is not self._graphs[0] else self._m0
                
                # 使用束搜索搜索此层的最佳邻居
                entry_points = self._search_base_layer(
                    new_point, entry_points, layer, ef, allow_soft_deleted=True
                )
                
                # 将新节点连接到此层的最佳邻居
                # 使用启发式剪枝选择多样化、高质量的邻居
                layer[key] = {
                    p: d
                    for d, p in self._heuristic_prune(
                        [(-mdist, p) for mdist, p in entry_points], level_m
                    )
                }
                
                # 更新现有邻居以包含新节点
                # 这维护了图的双向性质
                for neighbor_key, dist in layer[key].items():
                    # 将新节点添加到此邻居的连接列表中
                    # 并剪枝以维护最大邻居数
                    layer[neighbor_key] = {
                        p: d
                        for d, p in self._heuristic_prune(
                            [(d, p) for p, d in layer[neighbor_key].items()]
                            + [(dist, key)],  # 添加新节点
                            level_m,
                        )
                    }
                    
        # 阶段3：如果此节点比现有层级更高，创建新层级
        # 当我们获得非常高的层级分配时会发生这种情况（罕见但可能）
        for _ in range(len(self._graphs), level + 1):
            self._graphs.append(self._layer_class(key))
            # 新节点成为此新层级的入口点
            self._entry_point = key

    def _update(self, key: Hashable, new_point: np.ndarray, ef: int) -> None:
        """更新索引中与键关联的点。

        当更新一个已存在的点时，我们需要：
        1. 更新点的数据
        2. 重新计算并更新所有相关邻居的连接
        3. 修复受影响节点的连接以维护图质量

        这个算法确保当点的位置改变时，图的连接结构仍然保持最优，
        从而维持搜索的准确性和效率。

        Args:
            key (Hashable): 要更新的点的键。
            new_point (np.ndarray): 要更新到的新点。
            ef (int): 插入期间考虑的邻居数。

        Raises:
            KeyError: 如果键在索引中不存在。
        """
        # 验证键是否存在
        if key not in self._nodes:
            raise KeyError(key)
            
        # 更新点的数据
        self._nodes[key].point = new_point
        
        # 如果入口点是索引中的唯一点，我们不需要更新索引
        # 因为只有一个点，没有邻居需要重新连接
        if self._entry_point == key and len(self._nodes) == 1:
            return
            
        # 遍历所有包含此键的图层级
        for layer in self._graphs:
            if key not in layer:
                break  # 键在更高层级不存在
                
            # 确定此层的最大邻居数
            layer_m = self._m if layer is not self._graphs[0] else self._m0
            
            # 创建键的二度邻域中的点集合
            # 二度邻域包括：键本身 + 键的直接邻居 + 邻居的邻居
            # 这确保了我们在重新连接时考虑足够多的候选点
            neighborhood_keys = set([key])
            for p in layer[key].keys():
                neighborhood_keys.add(p)
                for p2 in layer[p].keys():
                    neighborhood_keys.add(p2)
                    
            # 对于键的每个邻居，我们将其与键的二度邻域中的前ef个邻居连接
            for p in layer[key].keys():
                # 使用优先队列来找到最近的候选邻居
                cands = []
                elem_to_keep = min(ef, len(neighborhood_keys) - 1)
                
                # 遍历二度邻域中的所有候选点
                for candidate_key in neighborhood_keys:
                    if candidate_key == p:
                        continue  # 跳过自己
                        
                    # 计算候选点与当前邻居的距离
                    dist = self._distance_func(
                        self._nodes[candidate_key].point, self._nodes[p].point
                    )
                    
                    # 维护一个大小为elem_to_keep的优先队列
                    if len(cands) < elem_to_keep:
                        heapq.heappush(cands, (-dist, candidate_key))
                    elif dist < -cands[0][0]:
                        # 如果新距离更小，替换队列中的最大距离
                        heapq.heappushpop(cands, (-dist, candidate_key))
                        
                # 使用启发式剪枝选择最终的邻居连接
                # 这确保邻居的多样性和质量
                layer[p] = {
                    p2: d2
                    for d2, p2 in self._heuristic_prune(
                        [(-md, p) for md, p in cands], layer_m
                    )
                }
                
        # 最后，修复被更新节点本身的连接
        # 这确保该节点连接到最佳的邻居
        self._repair_connections(key, new_point, ef)

    def _repair_connections(
        self,
        key: Hashable,
        new_point: np.ndarray,
        ef: int,
        key_to_delete: Optional[Hashable] = None,
    ) -> None:
        """修复指定节点的连接，确保其连接到最佳的邻居。

        这个方法在更新或删除节点后使用，用于重新建立该节点的连接。
        它模拟了插入算法的搜索过程，但只针对特定的节点进行连接修复。

        算法流程：
        1. 从最高层级开始，使用贪婪搜索找到良好的起始点
        2. 在每一层使用束搜索找到最佳邻居
        3. 使用启发式剪枝选择最终的邻居连接

        Args:
            key (Hashable): 要修复连接的节点的键。
            new_point (np.ndarray): 节点的新位置。
            ef (int): 搜索期间考虑的候选数。
            key_to_delete (Optional[Hashable]): 在删除操作中要排除的键。
        """
        # 从入口点开始搜索
        entry_point = self._entry_point
        entry_point_dist = self._distance_func(
            new_point, self._nodes[entry_point].point
        )
        entry_points = [(-entry_point_dist, entry_point)]
        
        # 从最高层级向下遍历所有图层级
        for layer in reversed(self._graphs):
            if key not in layer:
                # 如果键不在这一层，使用贪婪搜索找到最近的邻居
                # 这为下一层提供良好的起始点
                entry_point, entry_point_dist = self._search_ef1(
                    new_point,
                    entry_point,
                    entry_point_dist,
                    layer,
                    # 允许软删除的点作为入口点，因为我们需要遍历整个图
                    allow_soft_deleted=True,
                    key_to_hard_delete=key_to_delete,
                )
                entry_points = [(-entry_point_dist, entry_point)]
            else:
                # 如果键在这一层，使用束搜索找到最佳邻居
                level_m = self._m if layer is not self._graphs[0] else self._m0
                entry_points = self._search_base_layer(
                    new_point,
                    entry_points,
                    layer,
                    ef + 1,  # 加1以考虑节点本身
                    # 允许软删除的点作为入口点和邻居候选
                    allow_soft_deleted=True,
                    key_to_hard_delete=key_to_delete,
                )
                
                # 过滤掉被更新的节点本身
                # 节点不应该连接到自身
                filtered_candidates = [(-md, p) for md, p in entry_points if p != key]
                
                # 更新此层级上被更新节点的出边
                # 使用启发式剪枝确保邻居的多样性和质量
                layer[key] = {
                    p: d for d, p in self._heuristic_prune(filtered_candidates, level_m)
                }

    def query(
        self,
        query_point: np.ndarray,
        k: Optional[int] = None,
        ef: Optional[int] = None,
    ) -> List[Tuple[Hashable, float]]:
        """搜索查询点的k个最近邻。

        这实现了HNSW搜索算法，它包含两个阶段：
        1. 使用贪婪搜索从较高层级向下导航以找到良好的起始点
        2. 在基础层执行束搜索以找到k个最近邻

        该算法设计为比穷举搜索快得多，同时通过分层结构保持高精度。

        Args:
            query_point (np.ndarray): 要搜索的查询点。
            k (Optional[int]): 要返回的邻居数。如果为None，返回
                搜索期间找到的所有邻居。
            ef (Optional[int]): 搜索期间考虑的候选数。
                更高的值提供更好的精度但搜索更慢。如果为None，使用
                构建ef。

        Returns:
            List[Tuple[Hashable, float]]: 查询点的k个最近邻的
                (key, distance)对列表，按距离排序（最近的在前面）。

        Raises:
            ValueError: 如果索引为空（未找到入口点）。
        """
        # 如果未指定，使用构建ef
        if ef is None:
            ef = self._ef_construction
            
        # 检查索引是否为空
        if self._entry_point is None:
            raise ValueError("Entry point not found.")
            
        # 从入口点开始（最高层级节点）
        entry_point_dist = self._distance_func(
            query_point, self._nodes[self._entry_point].point
        )
        entry_point = self._entry_point
        
        # 阶段1：使用贪婪搜索从较高层级向下导航
        # 这快速缩小到基础层搜索的良好起始点
        # 我们跳过基础层（索引0），因为我们将在那里进行更彻底的搜索
        for layer in reversed(self._graphs[1:]):
            entry_point, entry_point_dist = self._search_ef1(
                query_point, entry_point, entry_point_dist, layer
            )
            
        # 阶段2：在基础层（第0层）执行束搜索
        # 这是我们进行彻底搜索以找到实际最近邻的地方
        candidates = self._search_base_layer(
            query_point, [(-entry_point_dist, entry_point)], self._graphs[0], ef
        )
        
        # 根据是否指定k处理结果
        if k is not None:
            # 仅返回k个最近邻
            candidates = heapq.nlargest(k, candidates)
        else:
            # 返回找到的所有候选，按距离排序
            candidates.sort(reverse=True)
            
        # 从内部格式（负距离，键）转换为用户格式（键，距离）
        return [(key, -mdist) for mdist, key in candidates]

    def _search_ef1(
        self,
        query_point: np.ndarray,
        entry_point: Hashable,
        entry_point_dist: float,
        layer: _Layer,
        allow_soft_deleted: bool = False,
        key_to_hard_delete: Optional[Hashable] = None,
    ) -> Tuple[Hashable, float]:
        """用于查找单个最近邻的贪婪搜索算法。

        这是一个简单的贪婪算法，从入口点开始，迭代地移动到最近的未访问邻居，
        直到找不到更近的邻居。它用于较高层级的导航，我们只需要为下一层找到
        良好的起始点。

        该算法维护候选的优先队列和已访问节点的集合以避免循环。

        Args:
            query_point (np.ndarray): 要搜索的查询点。
            entry_point (Hashable): 搜索的起始点。
            entry_point_dist (float): 从查询到入口点的距离。
            layer (_Layer): 要搜索的图层级。
            allow_soft_deleted (bool): 是否允许软删除的点作为结果返回。
            key_to_hard_delete (Optional[Hashable]): 永远不应返回的键
                （在删除操作期间使用）。

        Returns:
            Tuple[Hashable, float]: 表示找到的最近邻的(key, distance)元组。
        """
        # 使用入口点初始化
        candidates = [(entry_point_dist, entry_point)]
        visited = set([entry_point])
        best = entry_point
        best_dist = entry_point_dist
        
        while candidates:
            # 从优先队列中弹出最近的未访问节点
            dist, curr = heapq.heappop(candidates)
            
            # 早期终止：如果最近的候选比我们的最佳更远，
            # 我们找不到更好的了
            if dist > best_dist:
                break
                
            # 获取当前节点的所有未访问邻居
            neighbors = [p for p in layer[curr] if p not in visited]
            visited.update(neighbors)
            
            # 计算到所有邻居的距离
            dists = [
                self._distance_func(query_point, self._nodes[p].point)
                for p in neighbors
            ]
            
            # 处理每个邻居
            for p, d in zip(neighbors, dists):
                # 检查此邻居是否比我们当前的最佳更好
                if d < best_dist:
                    # 检查此邻居是否应该被排除
                    if (not allow_soft_deleted and self._nodes[p].is_deleted) or (
                        p == key_to_hard_delete
                    ):
                        # 跳过此邻居但继续探索其邻居
                        pass
                    else:
                        # 更新我们的最佳候选
                        best, best_dist = p, d
                        
                # 将此邻居添加到候选队列以进行进一步探索
                heapq.heappush(candidates, (d, p))
                
        return best, best_dist

    def _search_base_layer(
        self,
        query_point: np.ndarray,
        entry_points: List[Tuple[float, Hashable]],
        layer: _Layer,
        ef: int,
        allow_soft_deleted: bool = False,
        key_to_hard_delete: Optional[Hashable] = None,
    ) -> List[Tuple[float, Hashable]]:
        """在图层级中查找多个邻居的束搜索算法。

        这实现了"ef搜索"算法，维护一个包含迄今为止找到的最佳ef个候选的束。
        它用于基础层（第0层），我们需要找到多个高质量邻居，而不仅仅是单个最佳邻居。

        该算法使用两个数据结构：
        - candidates: 要探索的所有节点的优先队列
        - entry_points: 迄今为止找到的最佳ef个结果的优先队列

        Args:
            query_point (np.ndarray): 要搜索的查询点。
            entry_points (List[Tuple[float, Hashable]]): 初始候选，作为
                (-distance, key)对（负距离用于最大堆行为）。
            layer (_Layer): 要搜索的图层级。
            ef (int): 要维护的最大结果数（束宽度）。
            allow_soft_deleted (bool): 是否允许软删除的点作为结果返回。
            key_to_hard_delete (Optional[Hashable]): 永远不应返回的键
                （在删除操作期间使用）。

        Returns:
            List[Tuple[float, Hashable]]: 表示找到的最佳ef个邻居的
                (-distance, key)对列表，按距离排序（最近的在前面）。

        Note:
            输入的entry_points可能包含软删除的点，这取决于调用上下文。
            如果需要，调用者应该过滤这些点。
        """
        # 使用入口点初始化候选队列
        # 转换为(distance, key)格式以实现最小堆行为
        candidates = [(-mdist, p) for mdist, p in entry_points]
        heapq.heapify(candidates)

        # 跟踪已访问的节点以避免循环
        visited = set(p for _, p in entry_points)
        
        while candidates:
            # 从候选队列中弹出最近的未访问节点
            dist, curr_key = heapq.heappop(candidates)

            # 早期终止：如果最近的候选比我们最差的结果更远，
            # 我们无法进一步改善结果
            worst_dist = -entry_points[0][0]  # 当前结果集中最差(最大)距离
            # FIX: 之前的实现无论是否已达到ef都会提前终止，导致结果数量
            # 被错误限制在 ~m0 (+/- 少量)。只有在我们已经收集到 ef 个候选
            # 时才允许根据该条件提前终止。
            if dist > worst_dist and len(entry_points) >= ef:
                break
                
            # 获取当前节点的所有未访问邻居
            neighbors = [p for p in layer[curr_key] if p not in visited]
            visited.update(neighbors)
            
            # 计算到所有邻居的距离
            dists = [
                self._distance_func(query_point, self._nodes[p].point)
                for p in neighbors
            ]
            
            # 处理每个邻居
            for p, dist in zip(neighbors, dists):
                # 检查此邻居是否应该从结果中排除
                if (not allow_soft_deleted and self._nodes[p].is_deleted) or (
                    p == key_to_hard_delete
                ):
                    # 如果邻居被删除但仍然足够接近，探索其邻居
                    if dist <= worst_dist:
                        heapq.heappush(candidates, (dist, p))
                elif len(entry_points) < ef:
                    # 我们还没有找到足够的结果，添加此邻居
                    heapq.heappush(candidates, (dist, p))
                    heapq.heappush(entry_points, (-dist, p))  # 负值用于最大堆
                    worst_dist = -entry_points[0][0]  # 更新最差距离
                elif dist <= worst_dist:
                    # 我们有足够的结果，但此邻居比我们最差的更好
                    heapq.heappush(candidates, (dist, p))
                    # 用这个更好的结果替换最差的结果
                    heapq.heapreplace(entry_points, (-dist, p))
                    worst_dist = -entry_points[0][0]  # 更新最差距离

        return entry_points

    def _heuristic_prune(
        self, candidates: List[Tuple[float, Hashable]], max_size: int
    ) -> List[Tuple[float, Hashable]]:
        """剪枝候选以仅保留前max_size个多样化邻居。

        这实现了一个启发式剪枝算法，基于距离和多样性选择邻居。
        目标是避免有太多彼此非常接近的邻居，这会降低图的可导航性。

        算法工作原理：
        1. 按距离顺序处理候选（最近的在前）
        2. 对于每个候选，检查它是否与已选择的邻居太接近
        3. 仅保留保持良好多样性的候选

        这基于hnswlib的启发式剪枝算法：
        https://github.com/nmslib/hnswlib/blob/978f7137bc9555a1b61920f05d9d0d8252ca9169/hnswlib/hnswalg.h#L382

        Args:
            candidates (List[Tuple[float, Hashable]]): 表示潜在邻居的
                (distance, key)对列表，按距离排序。
            max_size (int): 要保留的最大邻居数。

        Returns:
            List[Tuple[float, Hashable]]: 表示保持良好多样性的
                剪枝邻居的(distance, key)对列表。
        """
        # 如果我们没有足够的候选，返回所有候选
        if len(candidates) < max_size:
            return candidates
            
        # 转换为堆以进行高效处理
        heapq.heapify(candidates)
        pruned = []
        
        while candidates and len(pruned) < max_size:
            # 获取最近的剩余候选
            candidate_dist, candidate_key = heapq.heappop(candidates)
            good = True
            
            # 检查此候选是否与任何已选择的邻居太接近
            for _, selected_key in pruned:
                # 计算此候选与已选择邻居之间的距离
                dist_to_selected_neighbor = self._distance_func(
                    self._nodes[selected_key].point, self._nodes[candidate_key].point
                )
                
                # 如果候选比到查询点更接近已选择的邻居，
                # 它不够多样化，我们拒绝它
                if dist_to_selected_neighbor < candidate_dist:
                    good = False
                    break
                    
            # 如果候选保持良好多样性，保留它
            if good:
                pruned.append((candidate_dist, candidate_key))
                
        return pruned

    def remove(
        self,
        key: Hashable,
        hard: bool = False,
        ef: Optional[int] = None,
    ) -> None:
        """使用软删除或硬删除从索引中移除点。

        这实现了基于hnswlib issue #4讨论的两种删除策略：
        https://github.com/nmslib/hnswlib/issues/4

        **软删除（默认）：**
        - 将点标记为已删除但保留在图中
        - 未来的查询不会返回此点
        - 没有新的边会指向此点
        - 该点仍用于图遍历（保持连通性）
        - 快速操作，但为已删除的点使用内存

        **硬删除：**
        - 完全移除点及其所有边
        - 使用与插入相同的算法修复受影响邻居的连接
        - 较慢的操作但释放内存
        - 通过重新连接邻居维护图质量

        在两种情况下，如果被删除的点是入口点，将从最高层选择新的入口点。

        Args:
            key (Hashable): 要移除的点的键。
            hard (bool): 如果为True，执行硬删除。否则，执行软删除。
            ef (Optional[int]): 重新分配期间考虑的候选数
                （仅用于硬删除）。如果为None，使用构建ef。

        Raises:
            KeyError: 如果键在索引中不存在。

        Example:
            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index.update({i: d for i, d in enumerate(data)})

                # 软删除一个点
                index.remove(0)
                print(0 in index)  # False

                # 硬删除同一个点
                index.remove(0, hard=True)

                # 清理所有软删除的点
                index.clean()
        """
        # 验证键存在
        if not self._nodes or key not in self._nodes:
            raise KeyError(key)
            
        # 如果我们正在删除当前入口点，处理入口点重新分配
        if self._entry_point == key:
            new_entry_point = None
            
            # 从最高层找到新的入口点
            for layer in reversed(list(self._graphs)):
                new_entry_point = next(
                    (p for p in layer if p != key and not self._nodes[p].is_deleted),
                    None,
                )
                if new_entry_point is not None:
                    break
                else:
                    # 此层在删除后将为空，移除它
                    self._graphs.pop()
                    
            if new_entry_point is None:
                # 这是索引中唯一的点，清空所有内容
                self.clear()
                return
                
            # 更新入口点
            self._entry_point = new_entry_point
            
        # 如果未指定，使用构建ef
        if ef is None:
            ef = self._ef_construction

        # 执行软删除（标记为已删除）
        self._nodes[key].is_deleted = True
        
        # 如果这只是软删除，我们就完成了
        if not hard:
            return

        # 执行硬删除（完全移除并修复连接）
        
        # 步骤1：找到所有具有指向被删除节点的出边的节点
        # 这些节点需要更新其连接
        keys_to_update = set()
        for layer in self._graphs:
            if key not in layer:
                break  # 节点在更高层不存在
            keys_to_update.update(layer.get_reverse_edges(key))
            
        # 步骤2：修复受影响节点的连接
        # 这将它们重新连接到其他好的邻居，维护图质量
        for key_to_update in keys_to_update:
            self._repair_connections(
                key_to_update,
                self._nodes[key_to_update].point,
                ef,
                key_to_delete=key,  # 不要将被删除的节点视为候选
            )
            
        # 步骤3：从所有图层中移除节点
        for layer in self._graphs:
            if key not in layer:
                break  # 节点在更高层不存在
            del layer[key]
            
        # 步骤4：从主存储中移除节点
        del self._nodes[key]

    def clean(self, ef: Optional[int] = None) -> None:
        """使用硬删除从索引中移除所有软删除的点。

        此方法对所有已被软删除的点执行硬删除，这释放内存并提高查询性能。
        当您有许多软删除的点时，这对于定期清理很有用。

        Args:
            ef (Optional[int]): 每个硬删除期间重新分配时考虑的邻居数。
                如果为None，使用构建ef。
        """
        # 找到所有软删除的键
        keys_to_remove = list(key for key in self._nodes if self._nodes[key].is_deleted)
        
        # 对每个软删除的点执行硬删除
        for key in keys_to_remove:
            self.remove(key, ef=ef, hard=True)

    def merge(self, other: HNSW) -> HNSW:
        """通过将当前索引与另一个索引合并来创建新索引。

        这创建一个包含两个索引中所有点的新HNSW索引。
        如果一个点存在于两个索引中（相同键），将使用另一个索引中的点
        （另一个索引优先）。

        新索引将具有与当前索引相同的参数和当前索引随机状态的副本。

        Args:
            other (HNSW): 要合并的另一个索引。

        Returns:
            HNSW: 包含两个索引中所有点的新索引。

        Example:
            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data1 = np.random.random_sample((1000, 10))
                data2 = np.random.random_sample((1000, 10))
                index1 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index2 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # 构建两个索引
                index1.update({i: d for i, d in enumerate(data1)})
                index2.update({i + len(data1): d for i, d in enumerate(data2)})

                # 合并到新索引中
                merged_index = index1.merge(index2)
        """
        # 创建当前索引的副本
        new_index = self.copy()
        
        # 添加另一个索引中的所有点（这将覆盖重复项）
        new_index.update(other)
        
        return new_index
