是 NetworkX 中用于计算**节点度中心性**的函数。度中心性是一种衡量节点在网络中的重要性的指标，它表示**节点的度数与可能的最大度数之比**。
度中心性的值介于 0 到 1 之间，越接近 1 表示节点在网络中的重要性越高。
**无向图**，度中心性的计算公式为：**节点的度中心性 = 节点的度数 / (节点总数 - 1)**。节点最大度数就是该节点与其他所有节点相连接！
有向图，度中心性的计算会有所不同。
```python
nx.degree_centrality(G)
```
- `G`：NetworkX 图对象，表示要计算节点度中心性的图。

```python
import networkx as nx

# 创建一个图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

# 计算节点的度中心性
degree_centrality = nx.degree_centrality(G)

# 打印结果
for node, centrality in degree_centrality.items():
    print("节点 {} 的度中心性: {:.3f}".format(node, centrality))
```

在上述示例中，我们创建了一个简单的图，并添加了一些边。然后，我们使用 `nx.degree_centrality()` 函数计算了图中每个节点的度中心性。最后，我们遍历度中心性字典，并打印每个节点的度中心性。

通过计算节点的度中心性，我们可以了解节点在网络中的连接程度，度中心性较高的节点可能在信息传播、影响传播等方面起着重要的作用。