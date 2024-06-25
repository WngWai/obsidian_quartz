`todense()` 是 NetworkX 中**稀疏矩阵对象**的一个方法，用于将稀疏矩阵转换为常规的密集矩阵。
```python
adj_matrix.todense()
```
- 无参数：`todense()` 方法不接受任何参数。
`todense()` 方法返回一个常规的adj_matrix.todense()，该矩阵是一个 **NumPy 数组**（`numpy.ndarray`）对象，表示稀疏矩阵的密集版本。

```python
import networkx as nx
import numpy as np

# 创建一个有向图
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# 计算图的邻接矩阵
adj_matrix = nx.adjacency_matrix(G)

# 将稀疏矩阵转换为常规矩阵
adj_matrix_dense = adj_matrix.todense()

# 打印常规矩阵
print(adj_matrix_dense)
```
输出：
```python
[[0 1 0]
 [0 0 1]
 [1 0 0]]
```

在这个示例中，我们首先创建了一个有向图，并使用 `nx.adjacency_matrix(G)` 计算了图的邻接矩阵。然后，我们使用 `todense()` 方法将稀疏矩阵转换为常规矩阵，并将其存储在 `adj_matrix_dense` 中。最后，我们打印了常规矩阵的结果。

常规矩阵是一个二维数组，其中的元素与稀疏矩阵中的元素对应。在上述输出中，常规矩阵的第一行表示节点1与节点2之间的连接（值为1），第二行表示节点2与节点3之间的连接（值为1），第三行表示节点3与节点1之间的连接（值为1）。对于没有连接的节点对，矩阵中的元素为0。
请注意，对于大型图和稀疏矩阵，将其转换为常规矩阵可能**会占用大量内存**，因此在处理大型数据时要谨慎使用。