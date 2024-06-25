在`sklearn.model_selection`模块中，`train_test_split()`函数用于将数据集划分为训练集和测试集。

**函数定义**：
```python
train_test_split(*arrays, **options)
```

**参数**：
- `*arrays`：一个或多个数组、列表或稀疏矩阵。它们将被划分为训练集和测试集。

- `test_size`（可选）：**测试集的大小**。可以是浮点数（表示比例）或整数（表示样本数）。默认值为0.25。

- `train_size`（可选）：训练集的大小。可以是浮点数（表示比例）或整数（表示样本数）。如果未指定，则默认为与`test_size`**互补**。

- `random_state`（可选）：**随机数种子**，用于控制随机划分的重现性。

- `shuffle`（可选）：布尔值，表示是否在划分之前对数据进行**洗牌**。默认为`True`。

- `stratify`（可选）：数组或类别标签，用于执行**分层抽样**。默认为`None`。

**返回值**：
函数返回一个划分后的**训练集和测试集元组**。

**示例**：
以下是使用`train_test_split()`函数划分数据集的示例：

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 输出划分结果
print("训练集样本数:", len(X_train))
print("测试集样本数:", len(X_test))
```

在上述示例中，我们首先从`sklearn.model_selection`模块导入`train_test_split`函数，并从`sklearn.datasets`模块加载了鸢尾花数据集。

然后，我们将特征数据存储在`X`中，将目标变量存储在`y`中。

接下来，我们使用`train_test_split()`函数将数据集划分为训练集和测试集。指定`test_size`参数为0.2，表示将20%的数据划分为测试集，其余80%作为训练集。`random_state`参数设置为42，以确保每次运行时划分结果相同。

最后，我们打印出划分后的训练集和测试集的样本数。

以下是打印出的内容示例：

```
训练集样本数: 120
测试集样本数: 30
```

在上述输出中，我们可以看到数据集被划分为120个训练样本和30个测试样本。

需要注意的是，`train_test_split()`函数可以用于任何类型的数据集划分，不仅仅局限于鸢尾花数据集。您可以根据自己的数据和需求进行适当的调整和使用。