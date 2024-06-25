是 NetworkX 中的一个函数，用于计算**图的平均最短路径长度**（average shortest path length）。
平均最短路径长度是指在一个图中，**任意两个节点之间的最短路径的平均长度**。它度量了图中节点之间的整体连接程度和距离。
```python
nx.average_shortest_path_length(G)
```

```python
import networkx as nx

# 创建一个图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

# 计算平均最短路径长度
avg_shortest_path_length = nx.average_shortest_path_length(G)

# 打印结果
print("平均最短路径长度：{:.3f}".format(avg_shortest_path_length))
```

在上述示例中，我们创建了一个简单的图，并添加了一些边。然后，我们使用 `nx.average_shortest_path_length(G)` 函数计算了图的平均最短路径长度。最后，我们打印了平均最短路径长度的结果。

平均最短路径长度是一个衡量图中节点之间距离的指标。较小的平均最短路径长度表示节点之间的连接较紧密，信息传播和影响传播的效率较高。较大的平均最短路径长度表示节点之间的连接较稀疏，信息传播和影响传播的效率较低。

需要注意的是，如果图不是连通的（存在孤立节点或多个连通分量），那么计算的平均最短路径长度将只针对图中的连通部分进行计算。