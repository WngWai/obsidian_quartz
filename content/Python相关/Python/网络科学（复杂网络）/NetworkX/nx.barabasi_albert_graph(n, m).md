是 NetworkX 中用于生成 **Barabási-Albert（BA） 随机图**的函数。该函数根据节点数 `n` 和每个新节点连到的现有节点数 `m`，生成一个具有**无标度特性**的随机图。
```python
nx.barabasi_albert_graph(n, m)
```

- `n`：整数，表示生成的图的**节点数量**。
- `m`：整数，表示**每个新节点连到的现有节点数**。

```python
import networkx as nx

# 生成 Barabási-Albert 随机图
n = 20  # 节点数
m = 2   # 每个新节点连到的现有节点数
ba_graph = nx.barabasi_albert_graph(n, m)

# 打印节点数量和边数量
print("节点数量:", ba_graph.number_of_nodes())
print("边数量:", ba_graph.number_of_edges())
```

在上述示例中，我们使用 `nx.barabasi_albert_graph()` 函数生成一个具有 20 个节点和每个新节点连到 2 个现有节点的 Barabási-Albert 随机图。然后，我们使用 `number_of_nodes()` 和 `number_of_edges()` 方法分别打印节点数量和边数量。

通过调整 `n` 和 `m` 参数的值，可以生成不同节点数量和每个新节点连到的现有节点数的 Barabási-Albert 随机图。Barabási-Albert 随机图具有**无标度特性**，即度**分布遵循幂律分布**，**节点的度数不均衡，少数节点具有大量的连接**。