是 NetworkX 库中用于计算图中边的数量的**方法**。它返回图对象 `G` 中**边的总数**。

```python
import networkx as nx

# 创建一个无向图
G = nx.Graph()

# 添加节点和边
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# 计算边的数量
num_edges = G.number_of_edges()
print(num_edges)  # 输出：3
```

在上述示例中，我们创建了一个无向图 `G`，并添加了三个节点和三条边。然后，我们使用 `G.number_of_edges()` 方法计算了图中边的数量，结果为 3。
