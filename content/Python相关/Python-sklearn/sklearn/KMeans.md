在`sklearn.cluster`模块中，`KMeans`类是用于K均值聚类的主要类。下面是`KMeans`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `cluster_centers_`：聚类中心的坐标。
2. `labels_`：每个样本点的标签，表示所属的聚类簇。
3. `inertia_`：每个样本点到其最近聚类中心的距离的总和，也称为簇内误差平方和（SSE）。

**方法**：

1. 模型拟合和预测：
   - `fit(X[, y, sample_weight])`：拟合K均值模型。
   - `predict(X)`：预测样本点所属的聚类簇。
   - `fit_predict(X[, y, sample_weight])`：拟合模型并返回样本点的聚类簇标签。
   
2. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。

以下是一个应用举例，展示如何使用`KMeans`类：

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成样本数据
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# 创建K均值聚类模型对象
model = KMeans(n_clusters=3, random_state=42)

# 拟合模型并得到样本点的聚类簇标签
labels = model.fit_predict(X)

# 获取聚类中心的坐标
centers = model.cluster_centers_

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', color='red')
plt.show()
```

在上述示例中，我们使用`make_blobs()`函数生成了一个具有3个簇的样本数据集，并将特征数据存储在`X`中，真实标签存储在`y`中。

然后，我们创建了一个`KMeans`的实例对象`model`，并指定了聚类簇的数量为3。

接下来，我们使用样本数据对模型进行拟合，并调用`fit_predict()`方法获取每个样本点的聚类簇标签。

我们可以使用`cluster_centers_`属性获取聚类中心的坐标。

最后，我们使用散点图可视化样本数据，并用不同颜色表示不同的聚类簇，同时使用红色的"X"标记表示聚类中心。

以上是`KMeans`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。