
## skewness()

^93ddaf

在R语言的e1071包中，`skewness()`函数用于计算**数据的偏度**（skewness）。
**函数定义**：
```R
skewness(x, na.rm = FALSE)
```
**参数**：
- `x`：要计算偏度的数据向量或数据框。
- `na.rm`：可选参数，用于指定是否在计算时删除缺失值。默认为`FALSE`，即不删除缺失值。
**示例**：
```R
library(e1071)

# 示例：计算数据偏度
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# 计算数据偏度
skew <- skewness(data)
print(skew)
```
在示例中，我们首先加载e1071包使用`library(e1071)`。然后，我们创建了一个数据向量`data`，其中包含了一组数值。

接下来，我们使用`skewness()`函数计算了数据向量`data`的偏度，并将结果赋值给`skew`变量。最后，我们打印计算得到的偏度值。

注意，`skewness()`函数计算数据的偏度，它度量了数据分布的不对称性。当偏度值为0时，数据分布呈现对称性；当偏度值大于0时，数据分布向右倾斜（右尾较长）；当偏度值小于0时，数据分布向左倾斜（左尾较长）。

#### 偏度介绍
偏度（skewness）是一种统计量，用于衡量数据分布的对称性或偏斜程度。它描述了数据分布在平均值附近的偏斜方向和程度。
常见的偏度计算方法是通过计算数据的三阶中心矩来获得。三阶中心矩是指数据值减去均值后的立方值的平均数。
下面是偏度的计算公式：
```
Skewness = (Sum((X - Mean)^3) / (N * StdDev^3))
```
其中：
- `X` 是每个数据点的值
- `Mean` 是数据的均值
- `StdDev` 是数据的标准差
- `N` 是数据点的数量

计算过程如下：
1. 计算数据的均值 `Mean` 和标准差 `StdDev`。
2. 对于每个数据点，计算 `(X - Mean)^3` 的立方值。
3. 将所有 `(X - Mean)^3` 的立方值相加得到总和。
4. 将总和除以 `(N * StdDev^3)` 得到偏度值。

根据计算结果的正负和大小，可以判断数据的偏斜方向和程度：
- 当偏度为0时，表示数据分布近似对称。
- 当偏度大于0时，表示数据右偏（正偏），即数据尾部向右延伸，右侧的极值较多。
- 当偏度小于0时，表示数据左偏（负偏），即数据尾部向左延伸，左侧的极值较多。

需要注意的是，偏度是对数据分布形态的一种度量，它并不提供关于数据分布形状的完整信息。因此，在分析数据时，还需要结合其他统计量和可视化方法来全面了解数据的特征。

## kurtosis()

^6c120d

在R语言中，`kurtosis()`函数是`e1071`包中的一个函数，用于计算数据的**峰度系数**。峰度系数是描述数据分布形态陡缓程度的统计量，其计算公式为样本的四阶矩除以样本的二阶矩的平方。

函数定义：`kurtosis(x, corrected = TRUE)`

参数介绍：

- `x`: 待计算峰度系数的数据。
- `corrected`: 是否进行修正，默认为TRUE。如果为TRUE，则计算峰度系数时进行样本标准化，即除以（n-1）和（n-2）的组合数，其中n为样本数量。如果为FALSE，则直接计算原始数据的峰度系数。

举例：

假设我们有一个名为data的数据框，其中包含一些数值数据。要计算这些数据的峰度系数，可以使用以下代码：

```r
r复制代码# 导入e1071包  library(e1071)    # 计算峰度系数  data <- c(2, 4, 6, 8, 10)  result <- kurtosis(data)  print(result)
```

输出结果为：

```
复制代码[1] 1.75
```

在上面的示例中，我们使用`kurtosis()`函数计算了数据`data`的峰度系数，并将结果存储在`result`变量中。最后，我们使用`print()`函数输出结果。由于数据`data`只有5个样本，因此计算出的峰度系数为1.75。



## svm()

svm()得到的是模型，超参数已经手动设置。可以用predict()函数预测苏分类变量！

`e1071`包中的`svm()`函数是用来构建支持向量机（SVM）模型的。SVM是一种强大的分类和回归算法，广泛用于各种机器学习任务。下面是`svm()`函数的定义、主要参数介绍以及综合应用举例。

```R
svm(formula, data, ..., subset, na.action = na.omit, scale = TRUE, type = NULL, kernel = "radial", degree = 3, gamma = if (is.vector(y)) 1 / ncol(x) else 1 / ncol(x[[1]]), coef0 = 0, cost = 1, nu = 0.5, class.weights = NULL, cachesize = 40, tolerance = 0.001, epsilon = 0.1, shrinking = TRUE, cross = 0, probability = FALSE, fitted = TRUE)
```

- **`formula`**：指定模型的公式，格式为`响应变量 ~ 解释变量`。
- **`data`**：用于训练模型的数据框。
- **`subset`**：用于指定训练样本的子集。
- **`na.action`**：指定缺失值的处理方式，默认是`na.omit`。

- **`scale`**：逻辑值，指示是**否对数据进行标准化**。默认是`TRUE`。

- **`type`**：指定SVM的**类型**。

	**C-classification**：适用于**一般的分类任务**。
	**nu-classification**：适用于分类任务，但需要**更细致控制**支持向量数量。
	
	**one-classification**：适用于**异常检测或孤立点检测**。
	
	**eps-regression**：适用于**一般的回归任务**，数据中有噪音时比较适用。
	**nu-regression**：适用于回归任务，但需要**更细致控制**支持向量数量

- **`kernel`**：指定核函数类型，如`linear`、`polynomial`（多项式）、`radial`高斯（径向）、`sigmoid`。默认是`radial`。

- **`degree`**：多项式核函数的度，默认是3。

- **`gamma`**：核函数中的**γ参数**。默认值是`1/ncol(x)`。

- **`coef0`**：核函数中的常数项，适用于多项式核和Sigmoid核。

- **`cost`**：C-分类SVM的**惩罚参数**，默认是1。

- **`nu`**：Nu-SVM和回归中的Nu参数，默认是0.5。
- **`class.weights`**：类别权重，用于不均衡类别的调整。
- **`cachesize`**：缓存大小，以MB为单位，默认是40。
- **`tolerance`**：终止准则的容忍度，默认是0.001。
- **`epsilon`**：在回归中使用的ε。
- **`shrinking`**：是否使用收缩启发式，默认是`TRUE`。
- **`cross`**：交叉验证折数，默认是0，表示不进行交叉验证。
- **`probability`**：是否计算概率估计，默认是`FALSE`。
- **`fitted`**：是否计算拟合值，默认是`TRUE`。


以下是一个综合应用示例，使用`iris`数据集来进行分类任务。
```R
library(e1071)

# 加载数据集
data(iris)

# 将数据集拆分为训练集和测试集
set.seed(123)
index <- sample(1:nrow(iris), 0.8 * nrow(iris))
train_data <- iris[index, ]
test_data <- iris[-index, ]

# 构建SVM模型
svm_model <- svm(Species ~ ., data = train_data, 
                 type = "C-classification", 
                 kernel = "radial", 
                 cost = 1, 
                 gamma = 1 / ncol(train_data))

# 打印模型
print(svm_model)

# 输出
Call:
svm(formula = Species ~ ., data = train_data, type = "C-classification", kernel = "radial", 
    cost = 1, gamma = 1/ncol(train_data))

Parameters:
   SVM-Type:  C-classification 
 SVM-Kernel:  radial 
       cost:  1 

Number of Support Vectors:  47



# 预测
pred <- predict(svm_model, newdata = test_data)

# 查看预测结果
table(Predicted = pred, Actual = test_data$Species)

# 计算混淆矩阵
confusionMatrix <- table(Predicted = pred, Actual = test_data$Species)
print(confusionMatrix)

# 计算准确率
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
print(paste("Accuracy:", accuracy))
```

解释
1. **数据准备**：首先加载`iris`数据集，并将其拆分为训练集和测试集。
2. **构建模型**：使用`svm()`函数构建SVM模型，指定类型为`C-classification`，核函数为`radial`，并设置成本（cost）和γ参数。
3. **模型预测**：使用训练好的模型对测试集进行预测。
4. **结果评估**：通过混淆矩阵和准确率来评估模型的性能。

总结
- **`svm()`函数**：用于构建和训练支持向量机模型。
- **主要参数**：包括模型公式、数据、核函数类型、惩罚参数、核函数参数等。
- **综合应用**：通过示例展示了如何使用`svm()`函数进行分类任务，包括数据准备、模型构建、预测和结果评估。

