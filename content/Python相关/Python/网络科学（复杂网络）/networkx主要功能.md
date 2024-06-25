[多亏学了这个python库，一晚上端掉了一个传销团伙。。。 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/565905529)

```python
import networkx as nx
```

[[G]]图

### 创建图
[[nx.Graph()]] 创建**无向图**，可以给边赋予权重（undirected Graph）
[[nx.DiGraph()]] 创建**有向图**，可以给边赋予权重（directed ）

[[nx.MultiGraph()]]  创建**多重无向图**，即两个结点之间的边数多于一条，又允许顶点通过同一条边和自己关联
[[nx.MultiDigraph()]] 创建**多重有向图**

### 生成随机图
[[nx.gnp_random_graph(n, p)]]随机无向图，**ER随机网络模型**
[[nx.random_regular_graph(d, n)]] **随机正则图**
[[nx.barabasi_albert_graph(n, m)]] **BA无标度网络图**
[[nx.watts_strogatz_graph(n, k, p)]] **WS小世界网络图**

### 图操作
[[G.add_node()、G.add_nodes_from()]] 添加**单个，多个节点**
[[G.add_edge()、G.add_edges_from()]] 添加**单条边，多条边**

[[nx.from_numpy_matrix()]] 根据**矩阵绘制图加权图**
[[G.add_weighted_edges_from()]] 添加带有**权重的边**

### 图分析
[[G.degree()]] 获得节点或图的**度**
[[nx.degree_histogram(G)]] 返回所有节点的**度分布序列**
[[nx.clustering(G)]] 计算**节点聚类系数**
[[nx.average_clustering(G)]] 计算**图中节点的平均聚类系数**

[[nx.shortest_path(G)]] 计算指定节点间的**最短路径**，分有权重和无权重
[[nx.average_shortest_path_length(G)]] **平均最短路径长度**
[[nx.diameter(G)]] 计算**图的直径**

[[G.number_of_nodes()]] 网络中**节点总数**
[[G.number_of_edges()]] 网络中**边的总数**

[[nx.degree_centrality(G)]] 计算**节点度的中心性**
[[in_degree_centrality(G)]]
[[out_degree_centrality(G)]] 

[[nx.degree_assortativity(G)]] 计算图的**度同配性**、度匹配性

[[nx.connected_components()]]查找无向图的**联通分量**，对于不连通的节点就过滤掉了？


### 图的可视化
[[nx.draw_networkx(G)]]  NetworkX自带的绘图函数，优先选择
[[nx.draw(G)]] 基于matplotlib库
[[nx.draw_networkx_edge_labels(G)]] 绘制边标签？？

