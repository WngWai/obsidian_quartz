用于获取图中**指定节点的邻居节点**的方法。它返回一个迭代器（在 NetworkX 2.x 版本中返回一个邻居节点视图对象），该迭代器包含与给定节点直接相连的邻居节点。下面是关于 `G.neighbors()` 方法的详细说明和示例：
- `G.neighbors(node)`：返回指定节点的邻居节点的迭代器或邻居节点视图对象。

```python
import networkx as nx

G = nx.Graph()

# 添加边
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(2, 4)

# 获取节点 2 的邻居节点
neighbors = G.neighbors(2)

print(list(neighbors))
```

在上述示例中，我们首先创建了一个空的无向图 `G`。然后，使用 `G.add_edge()` 方法添加了三条边，分别是 (1, 2)、(2, 3) 和 (2, 4)。接着，我们使用 `G.neighbors()` 方法获取节点 2 的邻居节点，并将其赋值给变量 `neighbors`。然后，我们将 `neighbors` 转换为列表并打印出来。

输出结果为：
```
[1, 3, 4]
```

可以看到，`G.neighbors(2)` 返回了与节点 2 直接相连的邻居节点的迭代器，即节点 1、节点 3 和节点 4。
请注意，将 `G.neighbors()` 的结果转换为列表是为了演示目的，而在实际应用中，通常直接使用迭代器或邻居节点视图对象进行遍历操作。
值得注意的是，在 NetworkX 2.x 版本中，`G.neighbors()` 方法返回一个邻居节点视图对象，它类似于迭代器，可以按需访问邻居节点。如果需要将邻居节点视图对象转换为列表，可以使用 `list()` 函数，如 `neighbors = list(G.neighbors(2))`。

