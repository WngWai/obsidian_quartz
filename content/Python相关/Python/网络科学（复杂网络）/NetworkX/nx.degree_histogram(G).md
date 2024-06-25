是 NetworkX 图对象的**方法**之一，用于计算图中**节点度数的直方图**。
直方图是一种统计图形，它将变量的值分成不同的区间，并计算每个区间内值的**频数或频率**。在这种情况下，`G.degree_histogram()` 返回一个**列表**，其中每个元素表示对应度数的节点数量。
```python
nx.degree_histogram(G)
```

G.degree_histogram()不行

```python
import networkx as nx

# 创建一个图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

# 计算度数直方图
## degree_histogram = G.degree_histogram()
degree_histogram = nx.degree_histogram(G)
# 打印结果
for i, count in enumerate(degree_histogram):
    print("度数 {}: {}".format(i, count))
```

在上述示例中，我们创建了一个简单的图，并添加了一些边。然后，我们使用 `G.degree_histogram()` 方法计算图中节点度数的直方图。最后，我们遍历直方图列表，并打印每个度数及其对应的节点数量。

请注意，`G.degree_histogram()` 返回的直方图列表中的索引表示度数，而列表中的值表示对应度数的节点数量。因此，索引 0 表示度数为 0 的节点数量，索引 1 表示度数为 1 的节点数量，以此类推。