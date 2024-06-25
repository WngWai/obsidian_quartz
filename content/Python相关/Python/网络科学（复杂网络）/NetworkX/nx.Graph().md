创建无向图的基本类。该类**没有参数**，可以通过添加**节点和边**来构建图。下面是`nx.Graph()`中参数的详细介绍和举例：
创建了一个名为`G`的**无向图**。
```python
import networkx as nx
G = nx.Graph()
```

关键字参数：
1. `name`：设置图的名称。
创建了一个名为"MyGraph"的无向图。
```python
import networkx as nx

G = nx.Graph(name="MyGraph")
```

2. `create_using`：指定用于创建图的**类或数据结构**。
使用有向图`H`作为创建无向图`G`的基础。
```python
import networkx as nx

H = nx.DiGraph()
G = nx.Graph([(1, 2), (2, 3), (3, 4)], create_using=H)
```

3. `node_container`：指定节点容器的类型。
我们使用`set`作为节点容器，以创建一个无向图`G`。
```python
import networkx as nx

G = nx.Graph(node_container=set)
```

4. `edge_container`：指定边容器的类型。
我们使用`dict`作为边容器，以创建一个无向图`G`。
```python
import networkx as nx

G = nx.Graph(edge_container=dict)
```

5. `multigraph_input`：指定是否接受多重边输入。
使用`multigraph_input=True`来允许接受重复的边输入，从而创建一个包含多重边的无向图`G`。
```python
import networkx as nx

G = nx.Graph([(1, 2), (1, 2)], multigraph_input=True)
```

这些是`nx.Graph()`中的关键字参数及其详细举例。关键字参数可根据需要使用，以设置图的属性和行为。请注意，并非所有关键字参数都适用于`nx.Graph()`，某些关键字参数可能适用于其他类型的图类（例如`nx.DiGraph`、`nx.MultiGraph`等）。

#### 列表来初始化图`G`的边
 `data`：可以是图的初始化数据。它可以是包含节点和边的列表、元组或其他类型的数据结构。
```python
G = nx.Graph([(1, 2), (2, 3), (3, 4)])
```
