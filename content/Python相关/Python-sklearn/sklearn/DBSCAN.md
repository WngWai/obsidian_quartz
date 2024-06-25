在`sklearn.cluster`模块中，`DBSCAN`类是用于密度聚类的主要类。下面是`DBSCAN`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `core_sample_indices_`：核心样本的索引。
2. `components_`：核心样本的坐标。
3. `labels_`：每个样本点的标签，表示所属的聚类簇。-1表示噪声点。

**方法**：

1. 模型拟合和预测：
   - `fit(X[, y, sample_weight])`：拟合DBSCAN模型。
   - `fit_predict(X[, y, sample_weight])`：拟合模型并返回样本点的聚类簇标签。
   - `predict(X)`：预测样本点所属的聚类簇。
   
2. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。

以下是一个应用举例，展示如何使用`DBSCAN`类：

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 生成样本数据
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# 创建DBSCAN模型对象
model = DBSCAN(eps=0.3, min_samples=5)

# 拟合模型并得到样本点的聚类簇标签
labels = model.fit_predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
```

在上述示例中，我们使用`make_moons()`函数生成了一个月亮形状的样本数据集，并将特征数据存储在`X`中，真实标签存储在`y`中。

然后，我们创建了一个`DBSCAN`的实例对象`model`，并指定了`eps`参数为0.3，`min_samples`参数为5。

接下来，我们使用样本数据对模型进行拟合，并调用`fit_predict()`方法获取每个样本点的聚类簇标签。

最后，我们使用散点图可视化样本数据，并使用不同颜色表示不同的聚类簇。

以上是`DBSCAN`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。