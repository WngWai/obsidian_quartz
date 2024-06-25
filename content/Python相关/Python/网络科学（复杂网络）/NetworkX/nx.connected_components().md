在 Python 中，`nx.connected_components()` 函数是 NetworkX 库中用于**找到无向图中的连通分量**的函数。NetworkX 是一个用于创建、分析和可视化复杂网络的库。

`nx.connected_components(G)` 返回的是一个**生成器对象**，它产生图 `G` 中的连接组件，每个组件表示为一个集合。**每个连接组件都是一个集合，包含了该组件中的所有节点**。这个集合对象不是单纯只有节点信息？还包括节点之间的连接属性吗？


以下是 `nx.connected_components()` 函数的基本信息：

**所属包：** NetworkX

**定义：**
```python
nx.connected_components(G)
```

**参数介绍：**
- `G`：一个 NetworkX 无向图对象。

**功能：**
找到无向图中的连通分量，返回一个迭代器，每个元素是一个包含图中节点的连通分量（集合）。

**举例：**
```python
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的无向图
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])

# 绘制图形
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=800)

# 查找连通分量，必须用list集合对了装入列表中
connected_components = list(nx.connected_components(G))
  
# 打印连通分量
print("Connected Components:", connected_components)

# 显示图形
plt.show()
```

**输出：**
```
Connected Components: [{1, 2, 3}, {4, 5, 6}]
```

![[Pasted image 20231215172055.png]]

在上述示例中，我们首先创建了一个简单的无向图 `G`，然后使用 `nx.connected_components(G)` 查找图中的连通分量。最后，我们将连通分量打印出来，并使用 Matplotlib 绘制了图形，其中不同的连通分量用不同的颜色表示。在这个例子中，图有两个连通分量：{1, 2, 3} 和 {4, 5, 6}。


### Connected Components
就是一个个子图，将子图连接的节点单独输出。
在图论中，一个无向图可以被**分解为若干个连通分量**（Connected Components）。一个连通分量是图中的一个最大的子图，其中任意两个节点都是通过边相连的，即从一个节点到另一个节点存在一条路径。

形象地说，连通分量就是图中的一个“岛屿”，每个岛屿上的节点都可以通过边相互到达，而不与其他岛屿相连。

连接分量的概念对于理解图的结构和组织很重要。在某些应用中，特别是社交网络、传播模型等领域，连接分量有时表示着独立的社群或子群。

例如，考虑下面的图：

```
1 -- 2    3 -- 4    5
```

这个图有三个连通分量：{1, 2}、{3, 4} 和 {5}。在每个连通分量中，节点之间都是相互连接的，但不同连通分量之间没有边。在 NetworkX 中，使用 `connected_components` 函数可以找到图中的所有连通分量。