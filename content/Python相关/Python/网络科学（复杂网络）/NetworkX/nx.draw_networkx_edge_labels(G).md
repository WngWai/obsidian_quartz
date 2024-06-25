 是 NetworkX 中用于在图形上**绘制边标签的函数**。它可以将边的属性值作为标签显示在相应的边上

**关键字参数说明：**
- `G`：图对象，可以是有向图或无向图。
- `pos`：节点的布局位置。它可以是一个字典，其中键是节点，值是节点的坐标。如果未提供，则使用默认布局算法。
- `edge_labels`：边标签的字典。字典的键是边的标识符（通常是一个元组），值是要显示的标签。这个参数是必需的。
- `font_size`：标签的字体大小。默认为 `10`。
- `font_color`：标签的颜色。可以是单个颜色值，也可以是包含标签颜色的列表或数组。
- `font_family`：标签的字体族。默认为 `'sans-serif'`。
- `font_weight`：标签的字体粗细。默认为 `'normal'`。
- `alpha`：标签的透明度。可以是介于 `0` 和 `1` 之间的值。默认为 `None`，表示不透明。
- `bbox`：标签的边框样式。可以是一个字典，包含边框样式的参数，如边界框填充（`pad`）、边框线宽（`lw`）等。
- `label_pos`：标签相对于边的位置。可以是一个浮点数，表示标签在边上的相对位置（`0` 表示起点，`1` 表示终点），或者是一个布尔值列表，指定是否将标签放在边的中间位置。默认为 `0.5`，表示标签放在边的中间位置。

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# 添加节点和边
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'C', weight=2)
G.add_edge('C', 'D', weight=1)

# 获取边权重属性
edge_weights = nx.get_edge_attributes(G, 'weight')

# 设置标签字体大小
font_size = 12

# 绘制图形
nx.draw_networkx(G, with_labels=True)

# 绘制边标签
nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=edge_weights, font_size=font_size)

# 显示图形
plt.show()
```

在这个示例中，我们创建了一个无向图 `G` 并添加了一些节点和边。每条边都带有一个名为 `'weight'` 的属性，表示边的权重。我们使用 `nx.get_edge_attributes()` 函数获取了边的权重属性，并将其存储在 `edge_weights` 字典中。

然后，我们设置了标签的字体大小为 `12`。接下来，我们使用 `nx.draw_networkx()` 绘制了图形，并使用 `nx.draw_networkx_edge_labels()` 绘制了边标签，将 `pos` 设置为节点的布局位置，`edge_labels` 设置为 `edge_weights` 字典，表示将边权重作为标签显示在相应的边上。
最后，我们使用 `plt.show()` 显示图形。

