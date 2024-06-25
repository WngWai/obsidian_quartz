在`sklearn.preprocessing`模块中，`StandardScaler`类是用于特征标准化的主要类。下面是`StandardScaler`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `scale_`：特征的缩放因子（标准差）。
2. `mean_`：特征的平均值。
3. `var_`：特征的方差。

**方法**：

1. 数据转换：
fit()计算特征的**平均值和标准差**。获得`StandardScaler`对象
transform()使用已计算的**平均值和标准差（即，`StandardScaler`对象）对特征进行标准化转换。
   
fit_transform()将上面两步合二为一了。计算特征的平均值和标准差，并对特征进行**标准化转换**。
   
   - `inverse_transform(X)`：将已标准化的特征转换回原始数据。
   
2. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。

以下是一个应用举例，展示如何使用`StandardScaler`类：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data

# 数据标准化
scaler = StandardScaler()  
scaler.fit(x_train)
s_x_train = scaler.transform(x_train)
s_x_test = scaler.transform(x_test)

# 或者直接
scaler = StandardScaler()  
s_x_train = scaler.fit_transform(x_train)

# 打印标准化后的特征
print(X_scaled)
```

在上述示例中，我们加载了鸢尾花数据集，并将特征数据存储在`X`中。

然后，我们创建了一个`StandardScaler`的实例对象`scaler`。

接下来，我们使用特征数据对`scaler`进行拟合并进行标准化转换，使用`fit_transform()`方法。

最后，我们打印出标准化后的特征数据`X_scaled`。

以上是`StandardScaler`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。