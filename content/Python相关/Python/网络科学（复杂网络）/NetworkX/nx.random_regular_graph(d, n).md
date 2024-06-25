是 NetworkX 中用于生成**随机正则图**的函数。该函数根据度数 `d` 和节点数 `n`，生成一个具有**随机分布的度数且每个节点都具有相同度数**的图。
```python
nx.random_regular_graph(d, n)
```

- `d`：整数，表示每个节点的度数。
- `n`：整数，表示生成的图的节点数量。

```python
import networkx as nx

# 生成随机正则图
d = 3  # 每个节点的度数
n = 10  # 节点数
regular_graph = nx.random_regular_graph(d, n)

# 打印节点数量和边数量
print("节点数量:", regular_graph.number_of_nodes())
print("边数量:", regular_graph.number_of_edges())
```

在上述示例中，我们使用 `nx.random_regular_graph()` 函数生成一个具有每个节点度数为 3、节点数量为 10 的随机正则图。然后，我们使用 `number_of_nodes()` 和 `number_of_edges()` 方法分别打印节点数量和边数量。

通过调整 `d` 和 `n` 参数的值，可以生成不同度数和节点数量的随机正则图。在随机正则图中，每个节点具有相同的度数，并且边的分布是随机的。