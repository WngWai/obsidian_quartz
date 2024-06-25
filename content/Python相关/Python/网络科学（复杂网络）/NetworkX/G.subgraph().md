在 NetworkX（网络分析的 Python 库）中，`G.subgraph()` 函数用于**获取图 `G` 的一个子图，该子图包含给定的一组节点**。

以下是 `G.subgraph()` 函数的基本信息：

**所属包：** NetworkX

**定义：**
```python
G.subgraph(nodes)
```

**参数介绍：**
- `G`：一个 NetworkX 图对象。
- `nodes`：要包含在子图中的节点列表。

**功能：**
返回原始图 `G` 的一个子图，该子图包含参数 `nodes` 中指定的节点。

**举例：**
```python
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个简单的图
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (5, 6)])

# 绘制原始图形
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', node_size=800)

# 选择一个子图包含特定的节点
nodes_to_include = [1, 2, 3]
subgraph = G.subgraph(nodes_to_include)

# 绘制子图
nx.draw(subgraph, pos, with_labels=True, font_weight='bold', node_color='orange', node_size=800)

# 显示图形
plt.show()
```

在这个例子中，我们首先创建了一个简单的无向图 `G`，然后使用 `G.subgraph(nodes_to_include)` 创建了一个包含特定节点的子图。最后，我们使用 Matplotlib 绘制了原始图和子图，其中子图中的节点用橙色表示。

**输出：**
![[Pasted image 20231215171821.png]]


![[Pasted image 20231215171811.png]]

在图中，左边是原始图，右边是包含节点 {1, 2, 3} 的子图。