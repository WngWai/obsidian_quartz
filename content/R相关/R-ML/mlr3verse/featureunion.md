按理来说po("learner")构建的学习器操作符应该也行，但现阶段运行时就是有问题！？？


### 数据处理过程详细示例

假设我们有一个简单的数据集 `X` 和两个学习器 `lrn1` 和 `lrn2`。它们的预测结果如下：

- 输入数据 `X`：
  ```
  X = [[1, 2], [3, 4], [5, 6]]
  ```

- `lrn1` 的输出预测：
  ```
  pred1 = [[0.1, 0.9], [0.4, 0.6], [0.3, 0.7]]
  ```

- `lrn2` 的输出预测：
  ```
  pred2 = [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]
  ```

通过 `po("featureunion")` 合并后：

- 合并后的特征集：
  ```
  combined = [[0.1, 0.9, 0.8, 0.2], [0.4, 0.6, 0.7, 0.3], [0.3, 0.7, 0.6, 0.4]]
  ```


你说得对，`po("featureunion")` 是用于合并多个学习器的输出。`po("featureunion")` 的作用是将多个特征集（即多个学习器的输出）合并成一个特征集。我们需要确保每个学习器的输出可以被 `po("featureunion")` 处理。

为了更加清晰地展示 `po("featureunion")` 的作用和工作流程，我们将以一个简单的示例来说明，包括数据如何通过多个学习器处理并被 `po("featureunion")` 合并。

### 示例数据和学习器

我们使用 iris 数据集作为示例，并使用 `classif.rpart` 和 `classif.ranger` 作为基础学习器。

```r
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3verse)

# 定义任务
data("iris")
task = TaskClassif$new(id = "iris", backend = iris, target = "Species")

# 定义基础学习器
lrn_rpart = lrn("classif.rpart")
lrn_ranger = lrn("classif.ranger")

# 定义管道节点
po_rpart = po("learner_cv", learner = lrn_rpart)
po_ranger = po("learner_cv", learner = lrn_ranger)

# 将基础学习器进行并行处理
po_stack = gunion(list(po_rpart, po_ranger)) %>>% po("featureunion")

# 定义元学习器
po_lrn_meta = po("learner", learner = lrn("classif.nnet", size = 4, decay = 0.1))

# 定义Stacking管道
pipeline_stacking = po_stack %>>% po_lrn_meta

# 定义图学习器
learner_stack = GraphLearner$new(pipeline_stacking)

# 定义重抽样对象
resampling = rsmp("cv", folds = 5)

# 得到重抽样结果
rr_stack = resample(task, learner_stack, resampling)

# 查看平衡准确率
bacc_stack = rr_stack$score(msr("classif.bacc"))
total_bacc_stack = rr_stack$aggregate(msr("classif.bacc"))

print(bacc_stack)
print(total_bacc_stack)
```

### 处理过程解析

1. **基础学习器定义**：
   - `po("learner_cv", learner = lrn_rpart)`：使用决策树学习器，交叉验证包装。
   - `po("learner_cv", learner = lrn_ranger)`：使用随机森林学习器，交叉验证包装。

2. **基础学习器并行处理**：
   - 使用 `gunion(list(po_rpart, po_ranger))` 并行处理两个学习器的输出。`gunion` 允许多个 `PipeOp` 并行处理数据并输出多个结果。

3. **合并输出**：
   - 使用 `po("featureunion")` 合并两个学习器的输出特征。`po("featureunion")` 的作用是将来自不同学习器的预测结果合并到一个单一的特征集。

4. **元学习器定义**：
   - 使用 `po("learner", learner = lrn("classif.nnet", size = 4, decay = 0.1))` 包装神经网络元学习器。

5. **堆叠管道定义**：
   - 将并行处理的学习器输出传递给元学习器，形成完整的堆叠管道。


### 代码中的操作

在上述代码中，通过 `po("featureunion")`，我们将 `classif.rpart` 和 `classif.ranger` 的输出预测结果合并为一个特征集，然后传递给元学习器进行最终预测。

这样可以确保每个基学习器的输出被正确处理并合并，形成最终的堆叠模型。