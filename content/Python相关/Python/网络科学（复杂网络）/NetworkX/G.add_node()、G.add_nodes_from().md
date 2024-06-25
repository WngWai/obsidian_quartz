### G.add_nodes()
方法用于向图中**添加节点**。它接受节点作为参数，并将其添加到图中。下面是 `G.add_node()` 方法的详细介绍和举例：

**参数说明：**
- `node_for_adding`：要添加的节点。

```python
import networkx as nx

G = nx.Graph()

# 添加单个节点
G.add_node(1)
G.add_node("A")

# 添加多个节点
nodes = [2, 3, 4]
G.add_nodes_from(nodes)

print(G.nodes())
```

在上述示例中，我们首先创建了一个空的无向图 `G`。然后，使用 `G.add_node()` 方法添加了单个节点 1 和节点 "A"，以及使用 `G.add_nodes_from()` 方法添加了节点 2、3 和 4。最后，我们使用 `G.nodes()` 方法打印出图中的所有节点。

输出结果为：

```
[1, 'A', 2, 3, 4]
```

可以看到，图 `G` 中成功添加了这些节点。

`node_for_adding` 参数可以是任何可哈希的对象，例如整数、字符串、元组等。您可以根据需要多次调用 `G.add_node()` 方法来添加多个节点。

请注意，如果向图中添加已经存在的节点，它们不会产生重复的节点。图数据结构会自动处理重复节点的情况，确保每个节点在图中只出现一次。

###  G.add_nodes_from()
是用于向图中**批量添加节点**的方法。它接受一个**可迭代对象** **列表、元组、集合**作为参数，该可迭代对象包含要添加到图中的节点

**参数说明：**
- `nodes_for_adding`：一个可迭代对象，包含要添加到图中的节点。
```python
import networkx as nx

G = nx.Graph()

# 添加多个节点
nodes = [1, 2, 3, 'A', 'B']
G.add_nodes_from(nodes)

print(G.nodes())
```

在上述示例中，我们首先创建了一个空的无向图 `G`。然后，使用 `G.add_nodes_from()` 方法将节点列表 `[1, 2, 3, 'A', 'B']` 添加到图中。最后，我们使用 `G.nodes()` 方法打印出图中的所有节点。

输出结果为：
```
[1, 2, 3, 'A', 'B']
```



#### G.add_node()和G.add_nodes()的区别
1. `G.add_node(node_for_adding)`：
   - 参数：`node_for_adding` 表示要添加的**单个节点**。
   - 功能：将给定的节点添加到图中。
   ````python
   import networkx as nx

   G = nx.Graph()
   G.add_node(1)
   G.add_node("A")
   ```

   在上述示例中，`G.add_node()` 方法分别将节点 1 和节点 "A" 添加到图 `G` 中。

   ````

2. `G.add_nodes(nodes_for_adding)`：
   - 参数：`nodes_for_adding` 是一个**可迭代对象**（如列表、元组等），包含要**添加到图中的多个节点**。
   - 功能：将给定的多个节点添加到图中。
   ````python
   import networkx as nx

   G = nx.Graph()
   nodes = [1, 2, 3]
   G.add_nodes(nodes)
   ```

   在上述示例中，`G.add_nodes()` 方法将节点列表 `[1, 2, 3]` 添加到图 `G` 中。

   ````
总结：
- `G.add_node()` 方法用于添加单个节点到图中。
- `G.add_nodes()` 方法用于添加多个节点到图中。

选择使用哪种方法取决于您要添加的节点数量和数据结构。如果只需要添加单个节点，使用 `G.add_node()` 方法更为简洁。如果要添加多个节点，可以使用 `G.add_nodes()` 方法并传入一个包含多个节点的可迭代对象。

#### G.add_nodes_from()和G.add_nodes()的区别
`G.add_nodes_from()`和`G.add_nodes()`是 NetworkX 中用于向图中添加节点的两种方法。

1. `G.add_nodes_from(nodes)`：
   - 参数：`nodes`是一个**可迭代对象**（如列表、元组、集合等），包含要添加到图中的节点。
   - 功能：将`nodes`中的所有节点添加到图中。

   ````python
   import networkx as nx

   G = nx.Graph()
   nodes = [1, 2, 3]
   G.add_nodes_from(nodes)
   ```

   在上述示例中，`G.add_nodes_from()`方法将节点列表`[1, 2, 3]`添加到图`G`中。

   ````
2. `G.add_nodes(*nodes)`：
   - 参数：`*nodes`是一个**可变数量**的参数，每个参数都是一个要添加到图中的节点。
   - 功能：将每个给定的节点添加到图中。
   ````python
   import networkx as nx

   G = nx.Graph()
   G.add_nodes(1, 2, 3)
   ```

   在上述示例中，`G.add_nodes()`方法分别将节点1、2和3添加到图`G`中。

   ````

总结：
- `G.add_nodes_from()`方法接受一个可迭代对象作为参数，并将该对象中的所有节点添加到图中。
- `G.add_nodes()`方法接受可变数量的参数，并将每个参数作为一个节点添加到图中。

使用哪种方法取决于您的数据结构和需求。如果您已经有一个节点列表或其他可迭代对象，`G.add_nodes_from()`是更方便的选择。如果您只有少量的节点，可以直接使用`G.add_nodes()`方法简洁地添加它们。