是 NetworkX 中的一个函数，用于获取图中**边的属性**。它可以返回一个**字典**，其中键是边的标识符（通常是一个元组），值是对应的属性值。
**参数说明：**
- `G`：图对象，可以是有向图或无向图。
- `name`：要获取的属性的名称。

```python
import networkx as nx

G = nx.Graph()

# 添加节点和边
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'C', weight=2)
G.add_edge('C', 'D', weight=1)

# 获取边的权重属性
edge_weights = nx.get_edge_attributes(G, 'weight')

print(edge_weights)
```

在这个示例中，我们创建了一个无向图 `G` 并添加了一些节点和边。每条边都带有一个名为 `'weight'` 的属性，表示边的权重。我们使用 `nx.get_edge_attributes()` 获取了图中边的权重属性，并将结果打印出来。

```
{('A', 'B'): 4, ('B', 'C'): 2, ('C', 'D'): 1}
```

这个字典表示了图中每条边的权重属性。
请注意，可以使用其他属性名称来获取不同的属性。您可以根据自己的需求自定义属性，并在 `nx.get_edge_attributes()` 中指定属性名称来获取相应的属性值。