在 `mlr3` 中，`rsmp("holdout")` 是一种用于将数据集划分为训练集和测试集的重采样方法。`holdout` 方法涉及几个参数的定义，主要是控制划分比例和相关设置。

```python
rsmp("holdout", ratio = 0.67)
```
1. **ratio**：
   - **描述**：训练集的比例。`ratio` 的值应该在 0 到 1 之间，表示数据集中用于训练的部分，其余部分则用于测试。
   - **默认值**：0.67
   - **设置方式**：`resampling$param_set$values$ratio = 0.7`

**~~stratify~~**：不行！！参数不对！
   - **描述**：是否进行分层采样。如果设置为 `TRUE`，则在划分数据集时会保持目标变量的类别分布。适用于分类任务。
   - **默认值**：FALSE
   - **设置方式**：`resampling$param_set$values$stratify = TRUE`
例如有三个类别，自然训练集和测试集中每个类别都应该按照对应的抽样比例来划分，例如数据中A（7：3），B（7：3），C（7：3），都是7:3的比例



### 示例

以下是一个完整的示例，展示如何使用 `rsmp("holdout")` 方法来划分数据集，并设置 `ratio` 和 `stratify` 参数。

```r
# 安装并加载必要的包
install.packages("mlr3")
install.packages("mlr3learners")
library(mlr3)
library(mlr3learners)

# 创建示例数据框
data <- iris

# 创建分类任务
task = TaskClassif$new(id = "iris", backend = data, target = "Species")

# 创建 holdout 划分方法
resampling = rsmp("holdout")

# 设置划分比例（例如 70% 训练集，30% 测试集）和分层采样
resampling$param_set$values$ratio = 0.7
resampling$param_set$values$stratify = TRUE

# 应用划分方法到任务
resampling$instantiate(task)

# 查看划分结果
print(resampling$train_set(1))
print(resampling$test_set(1))

# 创建逻辑回归学习器
learner = lrn("classif.log_reg")

# 训练模型
learner$train(task, row_ids = resampling$train_set(1))

# 进行预测
prediction = learner$predict(task, row_ids = resampling$test_set(1))

# 查看预测结果
print(prediction)

# 评估模型性能
accuracy = prediction$score(msr("classif.acc"))
cat(sprintf("Accuracy: %.2f\n", accuracy))
```

### 参数说明

1. **ratio**：在上面的示例中，我们将 `ratio` 设置为 0.7，这意味着 70% 的数据将用于训练，30% 的数据将用于测试。

   ```r
   resampling$param_set$values$ratio = 0.7
   ```

2. **stratify**：我们将 `stratify` 设置为 `TRUE`，这将确保在划分数据集时，目标变量 `Species` 的类别分布在训练集和测试集之间保持一致。

   ```r
   resampling$param_set$values$stratify = TRUE
   ```

### 总结

在 `mlr3` 中，`rsmp("holdout")` 是一种常用的重采样方法，用于将数据集划分为训练集和测试集。通过设置 `ratio` 和 `stratify` 参数，可以控制训练集的比例以及是否进行分层采样，确保目标变量的类别分布在训练集和测试集之间保持一致。通过这些设置，可以更好地进行模型的训练和评估。