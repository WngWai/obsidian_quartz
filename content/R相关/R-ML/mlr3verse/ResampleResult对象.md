`resample()` 函数返回的对象是一个 `ResampleResult` 对象，它包含了重采样的详细结果，包括每次重采样的性能评估、训练好的模型（如果 `store_models = TRUE`）、预测结果等。下面是 `ResampleResult` 对象的主要属性和方法，以及如何使用它们。

### `ResampleResult` 对象的主要属性

- **`task`**: 任务对象（`Task`），表示重采样使用的数据任务。
- **`learner`**: 学习器对象（`Learner`），表示用于重采样的学习器。
- **`resampling`**: 重采样对象（`Resampling`），表示重采样策略。

- **`predictions`**: 包含所有重采样迭代的预测结果的列表。

- **`resample_instance`**: 包含重采样实例的对象，记录每次重采样的训练和测试集划分。
- **`scores`**: 存储每次重采样的性能评估结果的数据框。
- **`models`**: 如果 `store_models = TRUE`，则存储每次重采样的训练好的模型。

### `ResampleResult` 对象的方法

- **`$aggregate(measure)`**: 计算重采样结果的**聚合性能指标**。
以分类来说，是整个模型的**平均值**！
- **`$score(measure)`**: 返回所有重采样**迭代的性能评估结果**。
以分类来说，是**每次迭代**的准确率！因为可能涉及类似K折验证的重抽样

- **`$prediction()`**: 返**回合并后**的所有重采样预测结果。
得到测试集中真实值和预测值的数据!
```R
# 定义softmax回归学习器
lrn_softmax = lrn("classif.multinom")
# 得到重抽样结果
rr_softmax = resample(task, lrn_softmax, resampling)
# 返回的就是测试集中真实值和预测值的数据!
rr_softmax$prediction()
```
- **`$predictions()`** 是返回每次迭代的预测值？


- **`$errors`**: 获取重采样过程中发生的错误（如果有）。
- **`$clone()`**: 创建 `ResampleResult` 对象的副本。

### 示例代码

下面是一个完整的示例，展示了如何使用 `ResampleResult` 对象的属性和方法。

#### 步骤 1: 安装和加载必要的包

```r
# 安装并加载必要的包

library(mlr3)
library(mlr3learners)
library(mlr3data)
```

#### 步骤 2: 创建任务、学习器和重采样策略

```r
# 加载内置的 iris 数据集
data("iris")

# 创建分类任务
task = TaskClassif$new(id = "iris", backend = iris, target = "Species")

# 创建决策树学习器
learner = lrn("classif.rpart")

# 定义交叉验证重采样策略（5 折交叉验证）
resampling = rsmp("cv", folds = 5)
```

#### 步骤 3: 执行重采样

```r
# 执行重采样
rr = resample(task, learner, resampling, store_models = TRUE)
```

#### 步骤 4: 使用 `ResampleResult` 对象

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

1. **打印 `ResampleResult` 对象**:
    ```r
    print(rr)
    ```
    简单打印 `ResampleResult` 对象，查看基本信息。

2. **聚合性能评估**:
    ```r
    acc = rr$aggregate(msr("classif.acc"))
    print(acc)
    ```
    计算和打印重采样的平均分类准确率。

3. **获取所有重采样迭代的性能评估结果**:
    ```r
    scores = rr$score(msr("classif.acc"))
    print(scores)
    ```
    获取和打印每次重采样的分类准确率。

4. **获取合并后的所有重采样预测结果**:
    ```r
    predictions = rr$prediction()
    print(predictions)
    ```
    获取和打印合并后的预测结果。

5. **检查重采样过程中是否有错误**:
    ```r
    errors = rr$errors