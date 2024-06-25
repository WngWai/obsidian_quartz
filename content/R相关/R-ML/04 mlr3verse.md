# 04 mlr3verse
## 概括
[官方文档](https://mlr-org.com/ecosystem.html)

![Pasted image 20240325105647.png](<Pasted image 20240325105647.png>)

mlr3种的三项注意内容：

- 表现形式
	
	**package** (highlighted in the first instance)
	
	**package::function() or function()** 非关键包中的函数，或易于混淆的函数用前者
	
	**$field** for fields (data encapsulated in an R6 class)
	
	**$method()** for methods (functions encapsulated in an R6 class) 调用类对象的方法
	
	**Class** (for R6 classes primarily, these can be distinguished from packages by context)

- 凡是机器学习中涉及到的数据集全部转换为data.table类型，取某列数据向量不用 **$** 符号。

- 机器学习中涉及赋值统一用 **=** 号，而非 **<-** 符号。

---
## 流程

- 待整理

	两个问题，一是暂时用的不多，二是数据包太新了，只能看英文，效率低。
	
	[ResultData](https://mlr3.mlr-org.com/reference/ResultData.html?q=Task#method-tasks-)
	
	[mlr3db数据库连接](mlr3db数据库连接.md) 操作后台数据库
	
	[mlr3filters特征选择](mlr3filters特征选择.md)特征值选择（如刨除无关特征）
	
	utilities其他工具，来绘制图形？

	[[mlr3torch深度学习]]

- 已整理
	
	[data.table数据转换包](data.table数据转换包.md)

	[mlr3任务创建、模型评估](mlr3任务创建、模型评估.md) 创建数据集任务，进行模型的比较和评估
	
	[mlr3learners监督学习包](mlr3learners监督学习包.md) 主要机器学习算法，偏向监督分析，用于对数据进行训练得到模型，并进行预测

	[mlr3measures模型性能评价包](mlr3measures模型性能评价包.md)
	
	[mlr3tuning超参数调优包](mlr3tuning超参数调优包.md) 超参数调试，在模型训练前使用，确定合适的超参数
	
	[mlr3viz可视化](mlr3viz可视化.md)可视化

- 进阶

	[mlr3pipelines管道](mlr3pipelines管道.md)特征工程，搭建图流学习器。链接不同算法，有点像深度学习神经网络


---
（待整理）

```python
# 安装和加载mlr3及相关包
install.packages("mlr3")
install.packages("mlr3learners")
install.packages("mlr3measures")
install.packages("mlr3tuning")
install.packages("mlr3pipelines")

library(mlr3)
library(mlr3learners)
library(mlr3measures)
library(mlr3tuning)
library(mlr3pipelines)

# 加载数据集
data(iris)
dataset <- as.data.table(iris)

# 定义任务
task <- TaskClassif$new("iris", backend = dataset, target = "Species")

# 定义学习器
learner <- lrn("classif.ranger")

# 训练模型
model <- learner$train(task)

# 评估模型性能
predictions <- model$predict(task)
measures <- msr("classif.ce")
result <- measures$score(task$truth(), predictions)

# 超参数调优
param_set <- ParamSet$new(list(
  ParamDbl$new("mtry", lower = 1, upper = 4),
  ParamInt$new("num.trees", lower = 100, upper = 1000)
))

tuner <- tnr("random_search", max_budget = 100)
resampling <- rsmp("holdout")
ctrl <- tnr("eval", resampling = resampling, measures = measures)

tuning_instance <- TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = resampling,
  measures = measures,
  tuner = tuner,
  terminator = trm("evals", n_evals = 100),
  control = ctrl
)

result <- tuning_instance$optimize(param_set)

# 输出最佳模型和最佳超参数
best_model <- result$learner
best_params <- result$learner$values
```




## 综合案例
好的，下面是一个综合的例子，展示如何结合 `mlr3db`, `mlr3filters`, `mlr3pipelines`, `mlr3`, `mlr3learners`, `mlr3measures`, `mlr3tuning`, 和 `mlr3viz` 等包进行一个完整的机器学习工作流。这个例子将涵盖数据加载、预处理、特征选择、模型训练、模型评估、超参数调优以及结果可视化。

### Iris 数据集分类

#### 1. 数据库连接与数据加载

首先，假设我们将 Iris 数据集存储在一个 SQLite 数据库中，我们将使用 `mlr3db` 包来连接数据库并加载数据。

```r
# 安装并加载必要的包
install.packages(c("DBI", "RSQLite", "mlr3", "mlr3db", "mlr3filters", "mlr3pipelines", "mlr3learners", "mlr3measures", "mlr3tuning", "mlr3viz"))
library(DBI)
library(RSQLite)
library(mlr3)
library(mlr3db)
library(mlr3filters)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3measures)
library(mlr3tuning)
library(mlr3viz)

# 创建与 SQLite 数据库的连接
conn <- dbConnect(RSQLite::SQLite(), dbname = ":memory:")

# 将 iris 数据集写入数据库
dbWriteTable(conn, "iris", iris)

# 定义数据源
data_source <- as_data_backend(conn, "iris")

# 创建任务
task <- TaskClassif$new(id = "iris_task", backend = data_source, target = "Species")
```

#### 2. 数据预处理和特征选择

使用 `mlr3pipelines` 创建一个数据预处理和特征选择的管道。

```r
# 创建特征选择过滤器
filter_variance <- flt("variance")
filter_anova <- flt("anova")

# 创建管道步骤
po_scale <- po("scale")
po_filter_var <- po("filter", filter = filter_variance, param_vals = list(filter.nfeat = 4))
po_filter_anova <- po("filter", filter = filter_anova, param_vals = list(filter.nfeat = 2))

# 构建管道：数据标准化 -> 方差过滤 -> ANOVA 过滤
pipeline <- po_scale %>>% po_filter_var %>>% po_filter_anova
```

#### 3. 模型训练与评估

选择一个学习器并构建训练和评估管道，使用 `mlr3learners` 包中的学习器。

```r
# 加载决策树分类器
learner <- lrn("classif.rpart")

# 将学习器添加到管道
pipeline <- pipeline %>>% po("learner", learner)

# 定义重采样策略
resampling <- rsmp("cv", folds = 3)

# 定义性能度量
measures <- msr("classif.acc")

# 执行重采样
resampling_result <- resample(task, pipeline, resampling, measures)
print(resampling_result$aggregate(measures))
```

#### 4. 超参数调优

使用 `mlr3tuning` 包进行超参数调优，以优化模型性能。

```r
# 定义搜索空间
search_space <- ps(
  classif.rpart.cp = p_dbl(lower = 0.001, upper = 0.1)
)

# 定义调优实例
instance <- TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = resampling,
  measure = measures,
  search_space = search_space,
  terminator = trm("evals", n_evals = 20)
)

# 定义调优算法
tuner <- tnr("grid_search", resolution = 10)

# 执行调优
tuner$optimize(instance)

# 获取最优超参数
print(instance$result)
```

#### 5. 结果可视化

使用 `mlr3viz` 包对结果进行可视化。

```r
# 加载 mlr3viz 包
library(mlr3viz)

# 绘制重采样结果
autoplot(resampling_result)

# 绘制调优结果
autoplot(instance)
```

### 各阶段的作用

1. **数据加载**：使用 `mlr3db` 从数据库中加载数据，并创建一个 `mlr3` 任务。这一步确保了数据的来源和准备。
   
2. **数据预处理**：通过 `mlr3pipelines`，对数据进行标准化处理，并应用特征选择过滤器（方差和 ANOVA）。这一步旨在清理和优化数据，为模型训练做准备。

3. **模型训练与评估**：结合 `mlr3learners` 和 `mlr3measures`，构建一个包含学习器的管道，并使用交叉验证进行评估。这一步旨在训练模型并评估其性能。

4. **超参数调优**：使用 `mlr3tuning`，定义调优实例和搜索空间，通过网格搜索优化超参数。调优可以进一步提升模型性能。

5. **结果可视化**：使用 `mlr3viz` 对重采样结果和调优结果进行可视化，帮助直观理解模型性能和超参数调优过程。

通过这个综合例子，你可以看到 `mlr3` 生态系统中的各个包是如何协同工作的，从数据加载到模型评估，每一步都可以通过模块化和灵活的方式进行。这种方法不仅提高了代码的可维护性和可读性，还使得整个机器学习工作流更为高效和直观。