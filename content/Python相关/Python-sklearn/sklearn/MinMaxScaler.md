在`sklearn.preprocessing`模块中，`MinMaxScaler`类用于特征缩放，将特征缩放到给定的最小值和最大值之间。下面是`MinMaxScaler`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `scale_`：每个特征的缩放因子。
2. `min_`：每个特征的最小值。
3. `data_min_`：每个特征的原始最小值。
4. `data_max_`：每个特征的原始最大值。

**方法**：

1. 数据转换：
   - `fit(X[, y])`：计算特征的最小值和最大值。
   - `transform(X)`：使用已计算的最小值和最大值对特征进行缩放转换。
   - `fit_transform(X[, y])`：计算特征的最小值和最大值，并对特征进行缩放转换。
   - `inverse_transform(X)`：将已缩放的特征转换回原始数据。
   
2. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。

以下是一个应用举例，展示如何使用`MinMaxScaler`类：

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data

# 创建MinMaxScaler对象
scaler = MinMaxScaler()

# 计算特征的最小值和最大值，并对特征进行缩放转换
X_scaled = scaler.fit_transform(X)

# 打印缩放后的特征
print(X_scaled)
```

在上述示例中，我们加载了鸢尾花数据集，并将特征数据存储在`X`中。

然后，我们创建了一个`MinMaxScaler`的实例对象`scaler`。

接下来，我们使用特征数据对`scaler`进行拟合并进行缩放转换，使用`fit_transform()`方法。

最后，我们打印出缩放后的特征数据`X_scaled`。

以上是`MinMaxScaler`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。