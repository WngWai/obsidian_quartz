```R
library(mlr3)
library(mlr3cluster)
task = TaskClust$new("usarrests", backend = USArrests)
task$task_type
```


`mlr3` 的 `TaskClust` 类是为了处理聚类任务而设计的。在机器学习中，聚类任务是指将数据集中的样本分组成若干个“簇”或“群组”，通常是基于样本之间的相似性。

### 属性：

1. **id**：任务的唯一标识符。
2. **backend**：一个 `DataBackend` 对象，用于存储任务相关的所有数据。
3. **col_roles**：一个命名列表（named list），包含了不同角色的列，比如特征（feature）、目标（target）等。
4. **ncol**：数据集中列的数量。
5. **nrow**：数据集中行的数量（样本数）。
6. **task_type**：任务类型，对于 `TaskClust`，这将是 `"clust"`。

### 主要方法：

#### 创建任务：

- **`$new()`**：构造函数，用于创建一个新的聚类任务实例。需要提供任务的ID和后端数据。

#### 数据访问：

- **`$data()`**：返回一个 `data.table`，包含所有的数据。
- **`$nrow()`**：返回数据集中样本的数量。
- **`$ncol()`**：返回数据集中特征的数量。
- **`$col_info()`**：返回一个 `data.table`，包含有关数据列的信息，如列角色和数据类型。

#### 数据修改：

- **`$select()`**：选择特定的特征列。
- **`$filter()`**：根据条件过滤样本。

#### 数据抽样：

- **`$sample()`**：执行数据抽样，可以指定抽样大小、替换、分层等选项。

#### 实用工具：

- **`$assert()`**：检查数据是否满足特定的要求。
- **`$summarize()`**：提供一个数据摘要，可以包括缺失值数量、特征类型等统计信息。

### 按功能分类：

1. **对象实例化**：
    - `TaskClust$new(id, backend, task_type = "clust")`

2. **数据访问和检索**：
    - `$nrow()`
    - `$ncol()`
    - `$data(cols = NULL, rows = NULL)`
    
3. **数据操作**：
    - `$select(selector)`
    - `$filter(expr)`
    
4. **抽样和分区**：
    - `$sample(size, replace = FALSE, prob = NULL, seed = NULL)`
    
5. **任务检查和信息摘要**：
    - `$assert(task_type, nrow, ncol)`
    - `$summarize()`

6. **元信息和设置**：
    - `$col_info()`：获取列的基本信息。
    - `$col_roles`：检查和设置列角色。

请注意，这里列出的不是 `TaskClust` 类的所有方法，而是一些主要的、按功能分类的方法。`mlr3` 是一个活跃的项目，可以通过参阅其文档或源代码获取更全面的方法列表和详细信息。