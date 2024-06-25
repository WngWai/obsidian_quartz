在`sklearn.tree`模块中，`DecisionTreeRegressor`类是用于回归问题的决策树模型。下面是`DecisionTreeRegressor`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `feature_importances_`：特征的重要性指标。
2. `max_features_`：在拟合过程中使用的最大特征数量。
3. `n_features_`：特征的数量。
4. `n_outputs_`：模型的输出数量。
5. `tree_`：训练后的决策树模型。

**方法**：

1. 模型拟合和预测：
   - `fit(X, y[, sample_weight, check_input, ...])`：拟合决策树回归模型。
   - `predict(X[, check_input])`：对给定的样本进行回归预测。
   - `apply(X[, check_input])`：返回每个样本所属的叶子节点的索引。
   
2. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。

以下是一个应用举例，展示如何使用`DecisionTreeRegressor`类：

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建DecisionTreeRegressor对象
model = DecisionTreeRegressor()

# 拟合决策树回归模型
model.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

在上述示例中，我们加载了波士顿房价数据集，并将特征数据存储在`X`中，目标变量存储在`y`中。

然后，我们使用`train_test_split()`函数将数据集划分为训练集和测试集。

接下来，我们创建了一个`DecisionTreeRegressor`的实例对象`model`。

然后，我们使用训练集对模型进行拟合，使用`fit()`方法。

最后，我们使用测试集进行预测，使用`predict()`方法，并计算预测结果的均方误差。

以上是`DecisionTreeRegressor`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。