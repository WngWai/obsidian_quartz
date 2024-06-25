 是 NetworkX 中用于向图对象添加带权重的边的方法。

以下是 `G.add_weighted_edges_from()` 的详细讲解：

- 参数：`G.add_weighted_edges_from(edges, weight='weight')`

  - `edges`：要添加的边的列表。每条边由一个包含两个节点和一个权重值的三元组表示。
  - `weight`：（可选）权重的属性名称。默认为 `'weight'`。

`G.add_weighted_edges_from()` 方法用于向给定的图对象 `G` 添加带权重的边。每条边由两个节点和一个权重值组成，通过一个包含这些三元组的列表进行传递。

以下是一个示例：

```python
import networkx as nx

# 创建一个空的有向图
G = nx.DiGraph()

# 添加带权重的边
edges = [(1, 2, 0.5),
         (2, 3, 0.8),
         (3, 1, 0.3)]

G.add_weighted_edges_from(edges)

# 打印图的边和权重
for u, v, w in G.edges(data='weight'):
    print(f"Edge: {u} -> {v}, Weight: {w}")
```

输出：

```
Edge: 1 -> 2, Weight: 0.5
Edge: 2 -> 3, Weight: 0.8
Edge: 3 -> 1, Weight: 0.3
```

在这个示例中，我们首先创建一个空的有向图 `G`。然后，我们使用 `G.add_weighted_edges_from()` 方法向图中添加带权重的边。`edges` 列表包含了三个元组，每个元组表示一条边，包含两个节点和一个权重值。最后，我们使用一个循环打印图的边和对应的权重。

`G.add_weighted_edges_from()` 方法使得可以方便地向图对象添加带权重的边，以构建具有权重信息的图结构。这在许多网络分析和算法中非常有用，例如最短路径算法、社区检测等。