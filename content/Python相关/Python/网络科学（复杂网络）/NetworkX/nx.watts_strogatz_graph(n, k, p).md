 是 NetworkX 中用于生成 **Watts-Strogatz 小世界图**的函数。该函数根据节点数 `n`、每个节点连接的最近邻节点数 `k` 和重新连边的概率 `p`，生成一个具有小世界特性的随机图。
```python
nx.watts_strogatz_graph(n, k, p)
```
- `n`：整数，表示生成的图的**节点数量**。
- `k`：整数，表示**每个节点连接的最近邻节点数**。对于偶数 `k`，每个节点将与其 `k/2` 个相邻节点连接到其右侧和左侧。对于奇数 `k`，每个节点将与其 `(k-1)/2` 个相邻节点连接到其右侧和左侧，以及一个节点连接到其右侧的下一个节点。
- `p`：浮点数，表示**重新连边的概率**。对于每条边 `(u, v)`，以概率 `p` 将其替换为连接节点 `u` 和 `w` 的边，其中 `w` 是随机选择的节点，不包括 `u` 和 `v`。

```python
import networkx as nx

# 生成 Watts-Strogatz 小世界图
n = 20  # 节点数
k = 4   # 每个节点连接的最近邻节点数
p = 0.2 # 重新连边的概率
ws_graph = nx.watts_strogatz_graph(n, k, p)

# 打印节点数量和边数量
print("节点数量:", ws_graph.number_of_nodes())
print("边数量:", ws_graph.number_of_edges())
```

在上述示例中，我们使用 `nx.watts_strogatz_graph()` 函数生成一个具有 20 个节点、每个节点连接的最近邻节点数为 4，以概率 0.2 重新连边的 Watts-Strogatz 小世界图。然后，我们使用 `number_of_nodes()` 和 `number_of_edges()` 方法分别打印节点数量和边数量。

通过调整 `n`、`k` 和 `p` 参数的值，可以生成不同节点数量、最近邻节点数和重新连边概率的 Watts-Strogatz 小世界图。该图具有短**平均路径长度**和**高聚类系数**，表现出小世界网络的特性。