在 `mlr3` 包中，`TaskRegr` 类是专门为回归任务设计的。它提供了一套方法和属性，用于处理、定义和操作回归分析任务。以下是对 `TaskRegr` 类的主要属性、方法的介绍，以及它们的应用实例。

### 主要属性

- **id (character):** 任务的唯一标识符。
- **backend (DataBackend):** 存储任务数据的后端。这通常是基于数据表的，比如 `DataBackendDataTable`。

格式得是data.frame、tibble、data.table等，在scale()标准后数据为"matrix" "array"，需要转化格式才行，as.data.frame、as_tibble、as.data.table！


- **target (character):** 目标变量的名称。这是模型训练中需要预测的变量。
- **nrow (integer):** 数据集中的行数。
- **ncol (integer):** 数据集中的列数（不包括目标变量）。
- **col_roles (named list):** 定义列角色（如特征、目标、权重等）的列表。

### 方法分类

#### 创建任务

- **new(id, backend, target, ...):** 创建一个新的回归任务。

#### 数据操作

- **filter(features):** 根据特征名筛选数据。
- **select(rows):** 根据行号选择数据。

#### 获取信息

- **head(n):** 获取数据集的前n行。
- **tail(n):** 获取数据集的后n行。
- **summarize():** 提供数据集的概要信息。

#### 任务操作

- **droplevels():** 如果目标变量是因子类型，删除不使用的水平。

### 应用举例

以下是一个应用示例，展示如何创建一个回归任务，并对其进行基本操作：

```r
library(mlr3)

# 以mtcars数据集为例
data("mtcars")

# 创建回归任务
task = TaskRegr$new(id = "mpg_task", backend = mtcars, target = "mpg")

# 查看任务概要
task$summarize()

# 获取前几行数据
print(task$head(5))

# 筛选特定的特征
task$filter(c("cyl", "disp", "hp"))

# 选择前10行数据
task$select(1:10)

# 查看修改后的任务概要
task$summarize()
```

这个例子展示了如何使用 `mlr3` 的 `TaskRegr` 类来构建一个回归任务，并对数据进行初步的探索和处理。通过定义回归任务，你可以清晰地指定哪个变量是预测目标，以及如何筛选和选择数据以适应你的分析需求。接下来，你可以利用 `mlr3` 的其他功能来训练和评估回归模型，进一步探索数据的特性和潜在的预测能力。