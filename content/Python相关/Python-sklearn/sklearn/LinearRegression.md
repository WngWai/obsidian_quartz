在`sklearn.linear_model`模块中，`LinearRegression`类是用于线性回归建模的主要类。下面是`LinearRegression`类的主要属性和方法的介绍，按照功能进行分类。

```python
LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
```
- `fit_intercept`：是否拟合截距，默认为True。如果设置为False，模型将不会拟合截距项。
- `normalize`：是否对输入数据进行归一化，默认为False。如果设置为True，则在拟合之前会对输入数据进行归一化处理。
- `copy_X`：是否复制输入数据，默认为True。如果设置为False，则在拟合过程中直接使用原始输入数据。
- `n_jobs`：并行运行的作业数，默认为None，表示使用单个处理器。


**属性**：

1. `coef_`：回归模型的系数（斜率）。
2. `intercept_`：回归模型的截距。
3. `rank_`：回归模型的系数矩阵的秩。
4. `singular_`：回归模型系数矩阵的奇异值。

**方法**：

   - `fit(X, y[, sample_weight])`：拟合线性回归模型。

   - `predict(X)`：基于输入数据进行预测。
   
3. 模型评估：
   - `score(X, y[, sample_weight])`：返回模型的拟合优度，即**R²系数**。
   
4. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。
   
以下是一个应用举例，展示如何使用`LinearRegression`类：

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载波士顿房价数据集
data = load_boston()
X = data.data
y = data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型对象
model = LinearRegression()

# 拟合模型
model.fit(X_train, y_train)

# 获取系数和截距
coefficients = model.coef_
intercept = model.intercept_

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)

# 打印结果
print("系数:", coefficients)
print("截距:", intercept)
print("均方误差:", mse)
```

在上述示例中，我们使用`load_boston()`函数加载波士顿房价数据集，并将特征数据存储在`X`中，目标变量存储在`y`中。

然后，我们使用`train_test_split()`函数将数据集划分为训练集和测试集。

接下来，我们创建了一个`LinearRegression`的实例对象`model`。

然后，我们使用训练集数据对模型进行拟合，即调用`fit()`方法。

我们可以使用`coef_`属性和`intercept_`属性获取模型的系数和截距。

然后，我们使用测试集数据进行预测，即调用`predict()`方法。

最后，我们使用`mean_squared_error()`函数计算预测结果与实际结果之间的均方误差，并打印出系数、截距和均方误差。

以上是`LinearRegression`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。