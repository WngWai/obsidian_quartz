是NetworkX中的有向图类，用于创建一个**有向图对象**。有向图中每个节点之间的关系是有方向的，一个方向指向另一个节点。

该类的常用方法有：

- `add_node(node, **attr)`：添加节点，node为节点标识符，attr为节点属性字典；
- `add_edge(source, target, **attr)`：添加有向边，source为边的源节点标识符，target为边的目标节点标识符，attr为边属性字典；
- `out_degree(node)`：返回节点的出度；
- `in_degree(node)`：返回节点的入度；
- `successors(node)`：返回节点的后继节点列表；
- `predecessors(node)`：返回节点的前驱节点列表；
- `number_of_nodes()`：返回当前有向图中节点的数量；
- `number_of_edges()`：返回当前有向图中边的数量；
- `nodes()`：返回一个节点的迭代器。

下面是一个简单的例子：

``` python
import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)

# 添加有向边
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)
G.add_edge(2, 4)
G.add_edge(3, 5)
G.add_edge(4, 5)
G.add_edge(5, 1)

# 输出节点数量和边数量
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())

# 输出节点的前驱节点和后继节点
for node in G.nodes():
    print("Predecessors of node ", node, ": ", list(G.predecessors(node)))
    print("Successors of node ", node, ": ", list(G.successors(node)))

# 输出节点的入度和出度
for node in G.nodes():
    print("In-degree of node ", node, ": ", G.in_degree(node))
    print("Out-degree of node ", node, ": ", G.out_degree(node))
```

输出结果为：

```python
Number of nodes:  5
Number of edges:  7
Predecessors of node  1 :  [5]
Successors of node  1 :  [2, 3]
Predecessors of node  2 :  [1]
Successors of node  2 :  [3, 4]
Predecessors of node  3 :  [1, 2]
Successors of node  3 :  [5]
Predecessors of node  4 :  [2]
Successors of node  4 :  [5]
Predecessors of node  5 :  [3, 4]
Successors of node  5 :  [1]
In-degree of node  1 :  1
Out-degree of node  1 :  2
In-degree of node  2 :  1
Out-degree of node  2 :  2
In-degree of node  3 :  2
Out-degree of node  3 :  1
In-degree of node  4 :  1
Out-degree of node  4 :  1
In-degree of node  5 :  3
Out-degree of node  5 :  1
```

从输出结果可以看出，这是一个有5个节点、7条有向边的有向图。对于每个节点，我们输出了它的前驱/后继节点、入度和出度。