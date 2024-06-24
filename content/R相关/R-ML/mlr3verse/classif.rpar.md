`mlr3` 包中的 `classif.rpart` 学习器是基于 `rpart` 包的**决策树分类器**。`rpart` 是 R 语言中的一个常用包，用于生成分类和回归树。下面是关于 `classif.rpart` 学习器的详细信息，包括函数定义、参数介绍以及综合应用举例。
- **所属包**: `mlr3learners`
- **基础包**: `rpart`

https://mlr3.mlr-org.com/reference/mlr_learners_classif.rpart

```r
learner = lrn("classif.rpart")
```


- **`minsplit`**: 节点进行分裂所需的最小观测数，默认值为 20。
- **`cp`**: **复杂度参数**，用于树剪枝，默认值为 0.01。
- **`maxcompete`**: 在选择最佳分裂时考虑的**竞争变量数**，默认值为 4。

- **`maxsurrogate`**: 用于处理缺失值的**替代分裂变量数**，默认值为 5。
- **`usesurrogate`**: 控制**是否使用替代分裂**，默认值为 2。

- **`xval`**: 用于**交叉验证的折数**，默认值为 10。

#### cp
- **类型**：`numeric`
- **默认值**：`0.01`
- **范围**：`[0, 1]`
- **说明**：复杂度参数。用来控制树的生长，防止过拟合。较高的值会使树更早地停止生长。

#### keep_model
- **类型**：`logical`
- **默认值**：`FALSE`
- **选项**：`TRUE, FALSE`
- **说明**：是否保留模型对象。

#### maxcompete
- **类型**：`integer`
- **默认值**：`4`
- **范围**：`[0, ∞)`
- **说明**：在每个分裂点显示的**竞争分裂数**。

#### maxdepth
- **类型**：`integer`
- **默认值**：`30`
- **范围**：`[1, 30]`
- **说明**：树的**最大深度**。限制树的深度可以防止过拟合。

#### maxsurrogate
- **类型**：`integer`
- **默认值**：`5`
- **范围**：`[0, ∞)`
- **说明**：每个分裂点**要保留的替代分裂数**。

#### minbucket
- **类型**：`integer`
- **默认值**：`-`
- **范围**：`[1, ∞)`
- **说明**：**终端节点**的**最小样本数量**。

#### minsplit
- **类型**：`integer`
- **默认值**：`20`
- **范围**：`[1, ∞)`
- **说明**：**分裂节点**所需的**最小样本数量**。

#### surrogatestyle
- **类型**：`integer`
- **默认值**：`0`
- **范围**：`[0, 1]`
- **说明**：控制替代分裂的选择方式。
    - `0`：按每个替代分裂的计数进行选择。
    - `1`：按类似度得分选择。

#### usesurrogate
- **类型**：`integer`
- **默认值**：`2`
- **范围**：`[0, 2]`
- **说明**：用于处理缺失值的方式。
    - `0`：不使用替代分裂。
    - `1`：仅使用好的替代分裂。
    - `2`：使用所有替代分裂。

#### xval
- **类型**：`integer`
- **默认值**：`10`
- **范围**：`[0, ∞)`
- **说明**：用于**剪枝的交叉验证折数**。如果为0，则不进行交叉验证。

综合例子：
以下是一个使用 `classif.rpart` 分类器的综合示例，展示如何在 `mlr3` 中使用这些参数，并进行模型训练和评估。

```r
# 加载必要的包
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3misc)
library(readr)

# 读取数据
cardata = read_csv('car.data', col_names = c('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'), show_col_types = FALSE)

# 数据编码
dummies <- dummyVars(class ~ ., data = cardata)
cardata_df  <- cbind(as.data.frame(predict(dummies, cardata)), class = factor(cardata$class))

# 转化为 data.frame 数据结构
cardata_table <- as.data.frame(cardata_df)

# 创建分类任务
task = TaskClassif$new(id = "car", backend = cardata_table, target = "class")

# 定义重抽样对象
set.seed(51040)
resampling = rsmp("cv", folds = 5)

# 定义决策树学习器并设置参数
lrn_rpart = lrn("classif.rpart",
                cp = 0.01,                # 复杂度参数
                maxdepth = 10,            # 树的最大深度
                minbucket = 5,            # 终端节点的最小样本数量
                minsplit = 20,            # 分裂节点所需的最小样本数量
                xval = 10                 # 用于剪枝的交叉验证折数
)

# 得到重抽样结果
rr_rpart = resample(task, lrn_rpart, resampling)

# 查看平衡准确率
acc_rpart = rr_rpart$score(msr("classif.bacc"))
total_acc_rpart = rr_rpart$aggregate(msr("classif.bacc"))

# 输出总的平衡准确率
print(total_acc_rpart)
```

代码解释：

1. **数据读取和预处理**：
   - 使用 `read_csv` 读取数据，并用 `dummyVars` 对数据进行编码，将类别特征转化为数值特征。
   - 使用 `as.data.frame` 将数据转化为 `data.frame` 结构。

2. **任务创建**：
   - 创建一个分类任务 `TaskClassif`，设置目标变量为 `class`。

3. **重抽样和评估**：
   - 使用 5 折交叉验证 (`cv`) 进行重抽样。
   - 定义决策树学习器 `classif.rpart`，并设置参数如 `cp`, `maxdepth`, `minbucket`, `minsplit`, 和 `xval`。

4. **计算和输出结果**：
   - 使用 `resample` 函数进行重抽样，并计算平衡准确率 (`classif.bacc`)。
   - 输出总的平衡准确率。

这个示例展示了如何使用 `mlr3` 包中的 `classif.rpart` 学习器解决分类问题，并通过设置不同的参数来优化模型的性能。


## 本分类器就对应的就是CART算法
`classif.rpart` 是 `mlr3` 包中基于 `rpart` 包的决策树分类器，主要用于分类任务。如果要进行回归任务，应该使用 `regr.rpart`。

### 分类与回归的设置

#### 分类任务
使用 `classif.rpart` 分类器解决分类问题。`rpart` 默认使用基尼系数作为分类的分裂标准。

#### 回归任务
使用 `regr.rpart` 分类器解决回归问题。`rpart` 默认使用最小化方差（mean squared error）作为回归的分裂标准。

### 使用基尼系数的分类问题

以下是一个使用 `classif.rpart` 分类器解决分类问题的示例，其中默认使用基尼系数作为分裂标准。

### 示例代码

```r
# 加载必要的包
library(mlr3)
library(mlr3learners)
library(readr)

# 读取数据
cardata = read_csv('car.data', col_names = c('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'), show_col_types = FALSE)

# 数据编码
dummies <- model.matrix(~ . + 0, data = cardata[ , -7])  # 将非目标变量编码为哑变量
cardata_df <- as.data.frame(cbind(dummies, class = cardata$class))

# 创建分类任务
task = TaskClassif$new(id = "car", backend = cardata_df, target = "class")

# 定义决策树学习器
learner = lrn("classif.rpart", cp = 0.01, maxdepth = 10, minbucket = 5, minsplit = 20, xval = 10)

# 定义重抽样对象
set.seed(51040)
resampling = rsmp("cv", folds = 5)

# 进行重抽样评估
rr = resample(task, learner, resampling)

# 查看评分
scores = rr$score(msr("classif.bacc"))
mean_score = rr$aggregate(msr("classif.bacc"))

# 输出总的平衡准确率
print(mean_score)
```

### 参数解释

- `cp`: 复杂度参数，用来控制树的生长，防止过拟合。较高的值会使树更早地停止生长。
- `maxdepth`: 树的最大深度，限制树的深度可以防止过拟合。
- `minbucket`: 终端节点的最小样本数量。
- `minsplit`: 分裂节点所需的最小样本数量。
- `xval`: 用于剪枝的交叉验证折数。

### 分类和回归的区别

#### 分类任务

- 使用 `classif.rpart`。
- 默认分裂标准是基尼系数（Gini index）。
- 目标变量是类别型（factor）。

#### 回归任务

- 使用 `regr.rpart`。
- 默认分裂标准是最小化方差（mean squared error）。
- 目标变量是连续型（numeric）。

### 示例代码 - 回归任务

```r
# 生成一些示例数据
set.seed(123)
n <- 100
x <- rnorm(n)
y <- x^2 + rnorm(n)
data <- data.frame(x = x, y = y)

# 创建回归任务
task_regr = TaskRegr$new(id = "example", backend = data, target = "y")

# 定义回归树学习器
learner_regr = lrn("regr.rpart", cp = 0.01, maxdepth = 10, minbucket = 5, minsplit = 20, xval = 10)

# 定义重抽样对象
resampling_regr = rsmp("cv", folds = 5)

# 进行重抽样评估
rr_regr = resample(task_regr, learner_regr, resampling_regr)

# 查看评分
scores_regr = rr_regr$score(msr("regr.mse"))
mean_score_regr = rr_regr$aggregate(msr("regr.mse"))

# 输出总的均方误差
print(mean_score_regr)
```

### 参数解释

与分类任务中的参数类似，回归任务的参数也用于控制树的生长和剪枝过程。

通过上述代码示例，可以看到如何在 `mlr3` 中使用 `classif.rpart` 和 `regr.rpart` 进行分类和回归任务的模型训练、评估和预测。分类问题默认使用基尼系数作为分裂标准，而回归问题默认使用最小化方差作为分裂标准。