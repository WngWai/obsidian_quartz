`mlr3` 中的 `Resampling` 类对象用于定义数据**重采样策略**，例如交叉验证、留出法等。它是一个非常灵活且强大的工具，可以与任务和学习器结合使用，以评估模型的性能。

### `Resampling` 类对象的主要属性和方法

#### 主要属性

- **`id`**: 重采样策略的唯一标识符（例如 `"cv"` 表示交叉验证）。
- **`param_set`**: 参数集合，包含该重采样策略的**所有可配置参数**。
- **`iters`**: 重采样迭代次数（例如，5 折交叉验证中的迭代次数为 5）。
- **`train_set`**: 返回给定迭代的训练集索引。
- **`test_set`**: 返回给定迭代的测试集索引。
- **`instance`**: 重采样实例，包含具体的训练集和测试集划分。

#### 主要方法

- **`$new(id, param_vals)`**: 创建新的 `Resampling` 对象。
- **`$instantiate(task)`**: **实例化重采样策略**，针对特定任务生成训练集和测试集划分。
- **`$train_set(i)`**: 返回第 `i` 次重采样的训练集索引。
- **`$test_set(i)`**: 返回第 `i` 次重采样的测试集索引。
- **`$iters`**: 返回重采样的迭代次数。
- **`$print()`**: 打印重采样策略的详细信息。

### 综合应用举例

下面是一个完整的示例，展示了如何使用 `Resampling` 类对象来定义和应用重采样策略。

#### 步骤 1: 安装和加载必要的包

```r
# 安装并加载必要的包
install.packages("mlr3")
install.packages("mlr3learners")
install.packages("mlr3data")

library(mlr3)
library(mlr3learners)
library(mlr3data)
```

#### 步骤 2: 创建任务

```r
# 加载内置的 iris 数据集
data("iris")

# 创建分类任务
task = TaskClassif$new(id = "iris", backend = iris, target = "Species")
```

#### 步骤 3: 创建学习器

```r
# 创建决策树学习器
learner = lrn("classif.rpart")
```

#### 步骤 4: 定义和实例化重采样策略

```r
# 定义交叉验证重采样策略（5 折交叉验证）
resampling = rsmp("cv", folds = 5)

# 实例化重采样策略
resampling$instantiate(task)
```

#### 步骤 5: 查看重采样策略的详细信息

```r
# 打印重采样策略的详细信息
print(resampling)

# 获取重采样的迭代次数
iters = resampling$iters
print(iters)

# 获取第一个迭代的训练集和测试集索引
train_set = resampling$train_set(1)
test_set = resampling$test_set(1)
print(train_set)
print(test_set)
```

#### 步骤 6: 执行重采样

```r
# 执行重采样
rr = resample(task, learner, resampling, store_models = TRUE)
```

#### 步骤 7: 使用 `ResampleResult` 对象

```r
# 查看重采样结果
print(rr)

# 获取平均性能指标
acc = rr$aggregate(msr("classif.acc"))
print(acc)

# 获取所有重采样迭代的性能评估结果
scores = rr$score(msr("classif.acc"))
print(scores)

# 获取合并后的所有重采样预测结果
predictions = rr$prediction()
print(predictions)

# 检查是否有错误
errors = rr$errors
print(errors)

# 获取存储的模型（如果 store_models = TRUE）
models = rr$models
print(models)

# 获取第一个重采样迭代的模型
first_model = models[[1]]
print(first_model)
```

### 详细解释

1. **定义重采样策略**:
    ```r
    resampling = rsmp("cv", folds = 5)
    ```
    创建一个 5 折交叉验证的重采样策略。

2. **实例化重采样策略**:
    ```r
    resampling$instantiate(task)
    ```
    针对特定任务