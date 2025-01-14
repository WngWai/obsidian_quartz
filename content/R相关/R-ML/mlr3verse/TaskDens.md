`mlr3` 是一个用于机器学习的 R 语言框架。`TaskDens` 类是 `mlr3` 中用于处理概率密度估计任务的类。概率密度估计是一种无监督学习任务，目的是估计给定数据的概率密度函数。

### TaskDens 类的属性：

- `id`：任务的唯一标识符。
- `task_type`：任务类型，对于 `TaskDens` 来说，值为 `"dens"`，表示概率密度估计任务。
- `backend`：数据存储的后端，可以是 `DataTable` 或 `data.frame` 等。
- `data_formats`：支持的数据格式名称。
- `feature_types`：特征的数据类型，例如数值或分类数据。
- `ncol`：数据集中的列数。
- `nrow`：数据集中的行数。
- `hash`：数据的哈希值，用于检测数据是否发生变化。

### TaskDens 类的主要方法：

按照功能分类，`TaskDens` 类的主要方法包括：

#### 创建和初始化：

- `TaskDens$new()`：创建一个新的 `TaskDens` 对象，初始化任务。

#### 数据处理：

- `filter()`：根据给定条件过滤数据。
- `select()`：选择数据集中的特定列。
- `head()`：获取数据集的前几行。
- `tail()`：获取数据集的后几行。
- `data()`：获取整个数据集或数据集的子集。

#### 信息获取：

- `task_type()`：获取任务类型。
- `nrow()`：获取数据集的行数。
- `ncol()`：获取数据集的列数。
- `feature_types()`：获取特征类型信息。

#### 实用功能：

- `formula()`：获取或设置与任务关联的公式。
- `levels()`：获取分类特征的水平。
- `droplevels()`：删除未使用的分类特征水平。

#### 操作和计算：

- `aggregate()`：使用指定的函数对数据进行聚合。
- `summarize()`：对数据集中的特定列进行总结。

请注意，这些方法的确切功能和可用性可能会根据 `mlr3` 包的版本和更新而有所不同。如果你需要更详细的信息，你应该查看 `mlr3` 的官方文档或使用 R 语言的帮助系统查询具体的方法。例如，你可以使用 `?TaskDens` 或 `help(TaskDens)` 来获取有关 `TaskDens` 类的帮助信息。