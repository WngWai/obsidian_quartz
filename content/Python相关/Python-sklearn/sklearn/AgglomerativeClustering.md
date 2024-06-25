在`sklearn.cluster`模块中，`AgglomerativeClustering`类是用于层次聚类的主要类。下面是`AgglomerativeClustering`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `labels_`：每个样本点的标签，表示所属的聚类簇。
2. `n_clusters_`：聚类簇的数量。
3. `n_leaves_`：层次聚类树的叶子节点数量。
4. `n_components_`：连接的组件数量。

`linkage`参数为`'single'`。可以模拟分裂式分层聚类，但不准确？


**方法**：

1. 模型拟合和预测：
   - `fit(X[, y])`：拟合层次聚类模型。
   - `fit_predict(X[, y])`：拟合模型并返回样本点的聚类簇标签。
   - `fit_transform(X[, y])`：拟合模型并返回转换后的数据。
   
2. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。

以下是一个应用举例，展示如何使用`AgglomerativeClustering`类：

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成样本数据
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# 创建层次聚类模型对象
model = AgglomerativeClustering(n_clusters=3)

# 拟合模型并得到样本点的聚类簇标签
labels = model.fit_predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
```

在上述示例中，我们使用`make_blobs()`函数生成了一个具有3个簇的样本数据集，并将特征数据存储在`X`中，真实标签存储在`y`中。

然后，我们创建了一个`AgglomerativeClustering`的实例对象`model`，并指定了聚类簇的数量为3。

接下来，我们使用样本数据对模型进行拟合，并调用`fit_predict()`方法获取每个样本点的聚类簇标签。

最后，我们使用散点图可视化样本数据，并用不同颜色表示不同的聚类簇。

以上是`AgglomerativeClustering`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。