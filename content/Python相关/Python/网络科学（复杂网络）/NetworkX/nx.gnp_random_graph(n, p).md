是 NetworkX 中用于生成 **Erdős-Rényi （ER）随机图**的函数。该函数根据节点数 `n` 和连接概率 `p` 生成一个具有**随机连接边的图**。
生成随机无向图最简单和常用的方法。

```python
nx.gnp_random_graph(n, p)
```

- `n`：整数，表示生成的图的节点数量。
- `p`：浮点数，表示连接每对节点的概率。对于每对节点，以概率 `p` 连接它们。


```python
import networkx as nx

# 生成 Erdős-Rényi 随机图
n = 10  # 节点数
p = 0.3  # 连接概率
random_graph = nx.gnp_random_graph(n, p)

# 打印节点数量和边数量
print("节点数量:", random_graph.number_of_nodes())
print("边数量:", random_graph.number_of_edges())
```

在上述示例中，我们使用 `nx.gnp_random_graph()` 函数生成一个具有 10 个节点和以概率 0.3 连接每对节点的 Erdős-Rényi 随机图。然后，我们使用 `number_of_nodes()` 和 `number_of_edges()` 方法分别打印节点数量和边数量。

通过调整 `n` 和 `p` 参数的值，您可以生成**不同节点数量和连接概率的 Erdős-Rényi 随机图**。请注意，随机图的边数可能会因为随机性而有所**变化**。