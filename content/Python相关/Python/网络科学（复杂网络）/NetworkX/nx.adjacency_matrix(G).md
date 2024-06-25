在NetworkX中，函数用于计算图`G`的**邻接矩阵**，但实际上的得到的是**稀疏矩阵**，需要转换为常规邻接矩阵。邻接矩阵是一个二维矩阵，用于表示图中节点之间的连接关系。
```python
nx.adjacency_matrix(G)
```
- `G`: 这是一个NetworkX图对象，可以是有向图（DiGraph）、无向图（Graph）或多重图（MultiGraph）。函数将计算该图的邻接矩阵。
`nx.adjacency_matrix(G)`函数返回一个稀疏的SciPy稀疏矩阵（`scipy.sparse.csr_matrix`格式），其中表示了图`G`的邻接矩阵。**稀疏矩阵**是一种用于表示稀疏数据（大多数元素为零）的高效数据结构。

```python
import networkx as nx
import scipy.sparse as sp

# 创建一个有向图
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# 计算图的邻接矩阵
adj_matrix = nx.adjacency_matrix(G)

# 将稀疏矩阵转换为常规矩阵
adj_matrix_dense = adj_matrix.todense()

# 打印邻接矩阵
print(adj_matrix_dense)
```
输出：
```PYTHON
[[0 1 0]
 [0 0 1]
 [1 0 0]]
```

在这个示例中，我们创建了一个有向图，并使用`nx.adjacency_matrix(G)`计算了图的邻接矩阵。然后，我们将稀疏矩阵转换为常规矩阵，并打印出邻接矩阵。

邻接矩阵的行和列对应于图中的节点，矩阵中的元素表示节点之间的连接关系。在上述输出中，邻接矩阵的第一行表示节点1与节点2之间的连接（值为1），第二行表示节点2与节点3之间的连接（值为1），第三行表示节点3与节点1之间的连接（值为1）。对于没有连接的节点对，矩阵中的元素为0。

通过使用`nx.adjacency_matrix(G)`，您可以获取图的邻接矩阵，进一步分析图的结构和节点之间的关系。请注意，对于大型图，邻接矩阵可能非常大，因此使用稀疏矩阵形式可以节省内存和计算资源。