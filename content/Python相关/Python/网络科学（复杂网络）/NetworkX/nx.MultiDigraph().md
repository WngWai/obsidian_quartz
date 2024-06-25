是 NetworkX 中的一个**类**，表示一个多重有向图（MultiDigraph）。多重有向图是一种允许存在多条相同节点之间的多条有向边的图结构。
在多重有向图中，每条边可以具有不同的属性，并且可以在同一对节点之间存在多条有向边。
以下是 `nx.MultiDigraph()` 类的一些主要特点：
- 允许存在多条相同节点之间的**多条有向边**。
- 每条边可以具有不同的属性。
- 支持节点和有向边的添加、删除和查询操作。
- 提供了一系列的图操作方法和算法。

下面是一个示例，展示如何使用 `nx.MultiDigraph()` 创建一个多重有向图：

```python
import networkx as nx

# 创建一个多重有向图
G = nx.MultiDigraph()

# 添加节点和有向边
G.add_node('A')
G.add_node('B')
G.add_edge('A', 'B', key=1)
G.add_edge('A', 'B', key=2)
G.add_edge('A', 'B', key=3)

# 打印图的信息
print(nx.info(G))
```

在这个示例中，我们使用 `nx.MultiDigraph()` 创建了一个多重有向图 `G`。然后，我们添加了两个节点 `'A'` 和 `'B'`，以及三条有向边，每条有向边都具有一个**键（key）** 属性。注意，这里的键是用于**区分同一对节点之间的不同有向边的标识符**。

最后，我们使用 `nx.info()` 打印了图 `G` 的信息，包括节点数、有向边数等。

输出结果如下：

```
Name: 
Type: MultiDiGraph
Number of nodes: 2
Number of edges: 3
Average in degree:   1.5000
Average out degree:   1.5000
```

这表明多重有向图 `G` 包含 2 个节点和 3 条有向边。

请注意，多重有向图与简单有向图在某些操作和算法上可能有所不同。在使用多重有向图时，请确保根据具体需求选择适当的方法和算法。