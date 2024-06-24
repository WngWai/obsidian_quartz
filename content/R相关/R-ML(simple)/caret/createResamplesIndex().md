```python

```


用于**创建数据集划分索引**的函数。它可以根据不同的策略(如随机、分层等)对数据集进行划分。

**函数定义**:
```r
createResamplesIndex(n, times = 1, p = 0.5, list = TRUE, strata = NULL, times.strata = NULL)
```

**参数介绍**:
- `n`: 数据集的行数。
- `times`: 指定创建多少个划分索引,默认为 1。
- `p`: 指定训练集的比例,取值范围为 (0, 1)。
- `list`: 逻辑值,决定是否返回列表形式的索引。默认为 `TRUE`。

- `strata`: 指定需要**分层的变量**,可以是**单个变量或变量向量**。

- `times.strata`: 指定分层时每个层次需要创建的索引个数。

**应用示例**:

假设我们有一个数据框 `df`:
```r
df <- data.frame(
  x = rnorm(100),
  y = sample(c("A", "B", "C"), 100, replace = TRUE)
)
```

我们想将数据集随机划分为 70% 的训练集和 30% 的测试集:

```r
library(caret)

# 创建随机划分的索引
set.seed(123)
index <- createResamplesIndex(nrow(df), p = 0.7)

# 根据索引划分训练集和测试集
train_data <- df[index$Train, ]
test_data <- df[index$Test, ]
```

如果我们想基于 `y` 变量进行分层采样:

```r
# 创建分层随机索引
set.seed(123)
index <- createResamplesIndex(nrow(df), p = 0.7, strata = df$y)

# 根据索引划分训练集和测试集
train_data <- df[index$Train, ]
test_data <- df[index$Test, ]
```

在这个例子中,我们指定 `strata = df$y` 对 `y` 变量进行分层,确保训练集和测试集中各个类别的比例相同。

总的来说,`createResamplesIndex()` 函数提供了灵活的数据集划分功能,可以满足各种复杂的数据划分需求。通过合理设置参数,可以实现随机、分层等不同的划分策略。