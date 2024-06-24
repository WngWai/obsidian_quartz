`mlr3`包中的`regr.nnet`学习器是基于神经网络的方法，用于回归任务。它主要使用`nnet`包中的`nnet`函数来实现。下面是对`regr.nnet`学习器的介绍、相关参数的介绍以及综合举例。

### 介绍

`regr.nnet`是一个用于回归任务的神经网络学习器。它适用于预测连续目标变量。该学习器可以处理各种特征类型，尤其适用于中小型数据集。

### 相关参数

`regr.nnet`的主要参数包括：

- `size`: 隐藏层的神经元个数，默认为3。
- `decay`: 权重衰减参数，用于防止过拟合，默认为0。
- `maxit`: 最大迭代次数，默认为100。
- `trace`: 是否打印训练过程信息，默认为TRUE。
- `MaxNWts`: 权重最大数量，默认为1000。
- `abstol`: 绝对误差容限，默认为1.0e-4。
- `reltol`: 相对误差容限，默认为1.0e-8。

### 综合举例

下面是一个使用`mlr3`中的`regr.nnet`学习器进行回归任务的示例。我们将使用`mtcars`数据集进行演示。

#### 环境配置

```r
library(mlr3)
library(mlr3learners)
library(mlr3verse)
library(data.table)
library(ggplot2)
```

#### 数据加载与预处理

```r
# 加载mtcars数据集
data("mtcars")
mtcars_dt <- as.data.table(mtcars)

# 创建回归任务
task = TaskRegr$new(id = "mtcars", backend = mtcars_dt, target = "mpg")
```

#### 定义学习器并设置参数

```r
# 定义regr.nnet学习器
learner = lrn("regr.nnet", size = 5, decay = 0.1, maxit = 200, trace = FALSE)
```

#### 评估学习器

```r
# 定义重抽样策略
resampling = rsmp("cv", folds = 5)

# 执行交叉验证
rr = resample(task, learner, resampling)

# 计算评估指标
rr$aggregate(msr("regr.mse"))
```

#### 模型训练和预测

```r
# 训练模型
learner$train(task)

# 进行预测
prediction = learner$predict(task)

# 打印预测结果
print(prediction$score(msr("regr.mse")))

# 可视化预测结果
results <- data.table(truth = task$truth(), response = prediction$response)
ggplot(results, aes(x = truth, y = response)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "Actual mpg", y = "Predicted mpg") +
  ggtitle("Actual vs Predicted mpg") +
  theme_light()
```

### 结果解释

在这个示例中，我们使用`mtcars`数据集，定义了一个包含5个隐藏层神经元、权重衰减为0.1、最大迭代次数为200的神经网络回归器。我们使用5折交叉验证来评估模型，并计算均方误差（MSE）。

通过这些步骤，我们可以方便地使用`mlr3`包中的`regr.nnet`学习器进行回归任务，并调整相关参数来优化模型性能。最后，我们还可视化了实际值和预测值的关系，以便更好地理解模型的表现。

希望这个示例能帮助您更好地理解和使用`mlr3`中的`regr.nnet`学习器。