在`sklearn.linear_model`模块中，`LogisticRegression`类是用于逻辑回归建模的主要类。下面是`LogisticRegression`类的主要属性和方法的介绍，按照功能进行分类。

**属性**：

1. `coef_`：逻辑回归模型的系数（权重）。
2. `intercept_`：逻辑回归模型的截距。
3. `n_iter_`：在求解逻辑回归模型时迭代的次数。

**方法**：

1. 模型拟合：
   - `fit(X, y[, sample_weight])`：拟合逻辑回归模型。
   
2. 模型预测：
   - `predict(X)`：基于输入数据进行预测。
   - `predict_proba(X)`：返回样本属于各个类别的概率。
   
3. 模型评估：
   - `score(X, y[, sample_weight])`：返回模型的准确率。
   
4. 模型属性获取：
   - `get_params([deep])`：获取模型的参数。
   - `set_params(**params)`：设置模型的参数。
   
以下是一个应用举例，展示如何使用`LogisticRegression`类：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型对象
model = LogisticRegression()

# 拟合模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 打印结果
print("准确率:", accuracy)
```

在上述示例中，我们使用`load_iris()`函数加载鸢尾花数据集，并将特征数据存储在`X`中，目标变量存储在`y`中。

然后，我们使用`train_test_split()`函数将数据集划分为训练集和测试集。

接下来，我们创建了一个`LogisticRegression`的实例对象`model`。

然后，我们使用训练集数据对模型进行拟合，即调用`fit()`方法。

然后，我们使用测试集数据进行预测，即调用`predict()`方法。

最后，我们使用`accuracy_score()`函数计算预测结果的准确率，并打印出准确率。

以上是`LogisticRegression`类的主要属性和方法介绍，以及一个简单的应用示例。您可以根据需要进一步调整和使用这些属性和方法。