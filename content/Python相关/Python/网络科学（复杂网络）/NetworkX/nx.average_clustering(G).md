是 NetworkX 中的一个函数，用于计算**图的平均聚集系数**（average clustering coefficient）。
平均聚集系数是指在一个图中，**节点的邻居之间实际存在的连接（边）占可能存在的连接（边）的比例的平均值**。它度量了图中节点集合的聚集程度或社团结构。
平均聚集系数是一种用于衡量网络中节点聚集性的指标，有助于理解网络的局部连接模式和社交或信息传播的特征。
```python
nx.average_clustering(G)
```

```python
import networkx as nx

# 创建一个图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

# 计算平均聚集系数
avg_clustering = nx.average_clustering(G)

# 打印结果
print("平均聚集系数：{:.3f}".format(avg_clustering))
```

在上述示例中，我们创建了一个简单的图，并添加了一些边。然后，我们使用 `nx.average_clustering(G)` 函数计算了图的平均聚集系数。最后，我们打印了平均聚集系数的结果。

平均聚集系数的取值范围在 0 到 1 之间。较高的平均聚集系数表示图中节点之间的连接更加密集，存在更多的社团结构或聚集性。较低的平均聚集系数表示图中节点之间的连接相对稀疏，社团结构相对较少。

