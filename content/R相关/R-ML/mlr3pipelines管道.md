`mlr3pipelines` 包是 `R` 语言中用于**构建预处理和模型训练管道**的工具。它允许用户创建复杂的机器学习工作流程，其中包括**数据预处理、特征工程、模型训练**等步骤。
用于构建和组合数据预处理、特征工程和模型训练的工作流。该包是 `mlr3` 生态系统的一部分，设计用于简化和模块化机器学习工作流。

1，Pipeline管道（简单的线性工作流）/Graph图（复杂的工作流）

多个PipeOps（管道操作）通过边（edges）连接而成

[[Graph类对象]]

"管道"（Pipeline）：管道是由一系列数据处理和模型训练步骤组成的工作流，适合描述简单的线性工作流，即数据处理和建模步骤按顺序执行。

"图"（Graph）：适合描述复杂的工作流，包括并行处理、分支和合并等情况。	

2，**管道操作（Pipeline Operator）**：
操作符是管道/图中的**基本单元**。

[[po()]] - 用于创建和检索“管道操作符”（Pipeline Operator）对象，这些对象可以将多个预处理步骤、模型和后处理步骤组合在一起。
   ```r
   # 创建一个管道操作符，结合特征选择和分类器
   pipeline = po("selector") %>>% lrn("classif.rpart")
   ```

**数据预处理操作**：
`po("scale")`：对数据进行**标准化**（均值为0，标准差为1）。
`po("impute")`：填补缺失值。用均值对缺失值进行**填补**？用中位数对缺失值进行填补？用众数对缺失值进行填补？
`po("encode")`：编码分类变量。

**特征工程操作**：
`po("pca")`：应用主成分分析 (PCA) 进行特征**降维**。
`po("filter")`：特征选择，应用**特征过滤**。

po("mutate") to add a new feature to the task 


**模型训练操作**：
po("learner")基本的管道操作节点，用于将一个学习器（如分类器或回归器）添加到管道中。它执行的任务通常是训练模型并预测数据。
`po("learner", lrn("classif.rpart"))`：决策树分类器。
`po("learner", lrn("classif.ranger"))`：随机森林分类器。

po("learner_cv")
`po("learner_cv"， lrn())` 是一个用于交叉验证的管道操作节点。它在管道中添加了交叉验证步骤，使得在训练过程中对模型进行交叉验证评估。

[[featureunion]]
`po("featureunion")` 是一个 `PipeOp`，用于并行地应用多个数据处理步骤，并**将它们的输出特征连接在一起**，形成一个**更大的特征向量/矩阵**。它类似于 scikit-learn 中的 `FeatureUnion`。



3，**图学习器（Graph Learner）**：

图学习器是由多个操作符和边组成的有向无环图（DAG），用于表示复杂的管道结构。图学习器是类似mlr3中的learner的，所以可以直接替换了使用！
   
[[GraphLearner类对象]]

GraphLearner$new()


待整理：
`gunion()` 是一个函数，用于创建多个并行的管道操作（graphs），然后将它们的输出合并到一个新的 `Graph` 对象中。

**PipeOpCbind**: 将两个数据集按列绑定，常用于合并特征或添加额外数据。
**`PipeOpFeatureUnion`**: 合并多个特征集合，**特征组合**。
**`PipeOpRemoveConstants`**: 移除常量特征，**特征选择**。
**PipeOpDateFeatures**: 从日期时间变量中提取特征，如年、月、日等。
**PipeOpTargetTrafo**: 对目标变量进行变换，比如对分类标签进行独热编码。

**控制流与数据流模块**：
**`PipeOpBranch`**: **创建**分支。
**`PipeOpUnbranch`**: **合并**分支。
**`PipeOpNOP`**: 不执行任何操作（占位符），**操作占位**。

### 示例1

```R
# 加载必要的包
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)

# 创建任务
task <- tsk("iris")

# 创建管道步骤
po_scale <- po("scale")
po_impute <- po("imputemean")
po_pca <- po("pca")
po_filter <- po("filter", filter = mlr3filters::flt("variance"))
po_learner <- po("learner", lrn("classif.rpart"))

# 创建管道
pipeline <- po_scale %>>% po_impute %>>% po_pca %>>% po_filter %>>% po_learner

# 定义重采样策略
resampling <- rsmp("cv", folds = 3)

# 定义性能度量
measures <- msr("classif.acc")

# 进行管道训练和评估
resample(task, pipeline, resampling, measures)
```

### 示例2
以下是一个简单的示例，展示如何在 `mlr3` 中使用管道操作进行数据预处理和模型训练：

```R
# 安装和加载必要的包
install.packages("mlr3")
install.packages("mlr3pipelines")
install.packages("mlr3learners")
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)

# 加载数据集
data("iris", package = "datasets")

# 创建分类任务
task = TaskClassif$new(id = "iris", backend = iris, target = "Species")

# 创建数据预处理操作符
scale_op = po("scale")  # 标准化数据
impute_op = po("impute")  # 填补缺失值

# 创建模型训练操作符
learner_op = po("learner", lrn("classif.rpart"))  # 决策树分类器

# 将操作符串联成管道
pipeline = scale_op %>>% impute_op %>>% learner_op

# 创建图学习器
graph_learner = GraphLearner$new(pipeline)

# 训练模型
graph_learner$train(task)

# 进行预测
prediction = graph_learner$predict(task)

# 评估模型性能
accuracy = prediction$score(msr("classif.acc"))
print(accuracy)
```

### 示例3

以下是一个更复杂的示例，展示如何在 `mlr3` 中使用管道操作进行特征选择和模型堆叠：

```R
# 创建特征选择操作符
filter_op = po("filter", filter = flt("anova"))

# 创建基础学习器
learner1 = po("learner", lrn("classif.rpart"))
learner2 = po("learner", lrn("classif.ranger"))

# 创建堆叠操作符
stack = gunion(list(learner1, learner2)) %>>% po("featureunion") %>>% po("learner", lrn("classif.log_reg"))

# 将操作符串联成管道
pipeline = filter_op %>>% stack

# 创建图学习器
graph_learner = GraphLearner$new(pipeline)

# 训练模型
graph_learner$train(task)

# 进行预测
prediction = graph_learner$predict(task)

# 评估模型性能
accuracy = prediction$score(msr("classif.acc"))
print(accuracy)
```


### po()和其他用法的差异
在 `mlr3` 框架中，`PipeOpScale$new()` 和 `po("scale")` 都用于创建标准化数据的管道操作符，但它们在使用方式上略有不同。

 `PipeOpScale$new()`

- **直接构造方法**：使用 `PipeOpScale$new()` 是通过显式调用构造函数来创建一个新的 `PipeOpScale` 对象。
- **灵活性**：这种方法允许在创建对象时传递更多的参数，具有更高的灵活性和可定制性。
- **示例**：

```r
library(mlr3pipelines)

# 通过构造函数创建 PipeOpScale 对象
pipeop_scale <- PipeOpScale$new()
```

 `po("scale")`

- **简洁方法**：使用 `po("scale")` 是一种快捷方式，利用 `po()` 函数来创建标准化数据的管道操作符。
- **简洁性**：这种方法更为简洁，并且在大多数情况下足够使用。
- **示例**：

```r
library(mlr3pipelines)

# 使用快捷方式创建 PipeOpScale 对象
pipeop_scale <- po("scale")
```

主要区别：
1. **创建方式**：
   - `PipeOpScale$new()`: 直接使用类的构造函数。
   - `po("scale")`: 使用简化函数 `po()`。

2. **参数传递**：
   - `PipeOpScale$new()`: 可以直接在构造函数中传递参数，适合需要进行更多配置的情况。
   - `po("scale")`: 更为简洁，不需要额外参数的情况下非常方便。

```r
# 安装并加载必要的包
install.packages("mlr3")
install.packages("mlr3pipelines")

library(mlr3)
library(mlr3pipelines)

# 创建一个分类任务
task <- TaskClassif$new(id = "iris", backend = iris, target = "Species")

# 使用 PipeOpScale$new() 创建标准化操作符
pipeop_scale_new <- PipeOpScale$new()
graph_new <- GraphLearner$new(pipeop_scale_new %>>% lrn("classif.rpart"))

# 使用 po("scale") 创建标准化操作符
pipeop_scale_po <- po("scale")
graph_po <- GraphLearner$new(pipeop_scale_po %>>% lrn("classif.rpart"))

# 训练和预测
graph_new$train(task)
predictions_new <- graph_new$predict(task)

graph_po$train(task)
predictions_po <- graph_po$predict(task)

# 打印结果
print(predictions_new)
print(predictions_po)
```

 总结：

- 使用 `PipeOpScale$new()` 时，可以直接访问和设置更多的参数，提供了更高的灵活性。
- 使用 `po("scale")` 时，提供了一种更为简洁的创建管道操作符的方法，适合快速应用标准操作符的情况。

选择哪种方法取决于具体需求和个人偏好。如果需要更多的配置选项，可以使用 `PipeOpScale$new()`；如果追求简洁，可以使用 `po("scale")`。


```r
# 加载必要的包
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(mlr3benchmark)

# 创建管道元素
scale_op <- PipeOpScale$new()
impute_op <- PipeOpImputeMean$new()
pca_op <- PipeOpPCA$new()
learner_op <- PipeOpLearner$new(learner = lrn("classif.rpart"))

# 创建管道
pipeline <- scale_op %>>% impute_op %>>% pca_op %>>% learner_op

# 创建任务
task <- tsk("iris")

# 将管道转换为学习器
graph_learner <- GraphLearner$new(pipeline)

# 定义评估度量
measure <- msr("classif.acc")

# 创建交叉验证实例
resampling <- rsmp("cv", folds = 3)

# 执行评估
rr <- resample(task, graph_learner, resampling, measure)
rr$aggregate(measure)
```

在这个例子中，我们创建了一个包含数据标准化、均值填补缺失值、主成分分析和决策树模型的管道，然后在 Iris 数据集上使用交叉验证进行评估。最终，我们输出了模型的分类准确率。
