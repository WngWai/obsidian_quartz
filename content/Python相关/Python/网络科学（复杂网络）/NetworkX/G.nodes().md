是一个**方法**而不是一个函数，它用于返回**图中的节点列表**。该方法**没有参数**
下面是使用`G.nodes()`方法返回图中的节点列表的示例：

```python
import networkx as nx

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 'A', 'B'])

nodes = G.nodes()
print(nodes)
```

在上述示例中，我们首先创建了一个无向图`G`，并使用`G.add_nodes_from()`方法添加了节点。然后，我们使用`G.nodes()`方法获取图中的所有节点，并将结果赋给变量`nodes`。最后，我们打印出`nodes`，即图中的节点列表。

输出结果为：

```
[1, 2, 3, 'A', 'B']
```

可以看到，`G.nodes()`方法返回了图中的所有节点。

`G.nodes()`方法返回一个节点视图（NodeView）对象，它类似于一个集合或列表，可以进行迭代和其他操作。您可以使用`list()`函数将节点视图转换为列表，以便进行进一步的处理。

```python
nodes = list(G.nodes())
```

希望这可以帮助您理解`G.nodes()`方法的参数和用法。