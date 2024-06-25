是 NetworkX 库中用于计算图中节点数量的**方法**。它返回图对象 `G` 中**节点的总数**。


```python
import networkx as nx

# 创建一个无向图
G = nx.Graph()

# 添加节点和边
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# 计算节点数量
num_nodes = G.number_of_nodes()
print(num_nodes)  # 输出：3
```

在上述示例中，我们创建了一个无向图 `G`，并添加了三个节点和三条边。然后，我们使用 `G.number_of_nodes()` 方法计算了图中节点的数量，结果为 3。

这个方法对于分析网络图的**大小、节点规模**以及计算相关**统计指标**时非常有用。