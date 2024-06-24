在 `mlr3` 中，`classif.svm` 学习器是用于分类任务的支持向量机（SVM）模型。该学习器基于 `e1071` 包中的 `svm` 函数。下面是 `classif.svm` 学习器的一些主要参数及其说明，之后会给出一个完整的示例来展示如何使用这些参数。

```R
learner = lrn("classif.svm",
              type = "C-classification",
              kernel = "radial",
              gamma = 0.1,
              cost = 10,
              scale = TRUE)
```

- **`type`**: SVM的**类型**。对于分类问题，通常设置为 `C-classification`。

	**C-classification**：适用于**一般的分类任务**。
	**nu-classification**：适用于分类任务，但需要**更细致控制**支持向量数量。
	
	**one-classification**：适用于**异常检测或孤立点检测**。
	
	**eps-regression**：适用于**一般的回归任务**，数据中有噪音时比较适用。
	**nu-regression**：适用于回归任务，但需要**更细致控制**支持向量数

- **`kernel`**: 核函数的**类型**，可以是 `linear`、`polynomial`（多项式）、`radial`（高斯）、`sigmoid` 等。默认是 `radial`。

- **`degree`**: **多项式核函数的度数**。仅在 `kernel = polynomial` 时使用。

- **`gamma`**: 核函数的**参数**。对于 `radial`、`polynomial` 和 `sigmoid` 核函数，`gamma` 参数是必须的。

- **`cost`**: 惩罚系数C，控制误差项与间隔的权衡。即模型对误分类的容忍度。
- **`coef0`**: 核函数中的独立项，主要用于 `polynomial` 和 `sigmoid` 核函数。
- **`scale`**: 是否对数据进行归一化。默认是 `TRUE`。

### 示例代码

以下是一个完整的示例，展示了如何使用 `classif.svm` 学习器，包括设置参数、训练模型、进行预测和评估模型性能。

```r
# 安装并加载必要的包
install.packages("mlr3")
install.packages("mlr3learners")
install.packages("mlr3data")
library(mlr3)
library(mlr3learners)
library(mlr3data)

# 加载数据集
data("car_eval")

# 创建任务
task = TaskClassif$new(id = "car_eval", backend = car_eval, target = "class")

# 使用分层交叉验证
split = rsmp("cv", folds = 5, stratify = TRUE)
split$instantiate(task)

# 获取训练集和测试集索引
train_set = split$train_set(1)
test_set = split$test_set(1)

# 创建支持向量机学习器并设置相关参数
learner = lrn("classif.svm",
              type = "C-classification",
              kernel = "radial",
              gamma = 0.1,
              cost = 10,
              scale = TRUE)

# 训练模型
learner$train(task, row_ids = train_set)

# 在测试集上进行预测
prediction = learner$predict(task, row_ids = test_set)

# 评估预测结果
performance = prediction$score(msr("classif.acc"))
print(performance)

# 输出预测结果和混淆矩阵
print(prediction)
print(prediction$confusion)
```

### 参数调整与优化

在实际应用中，可以使用 `mlr3tuning` 包来进行参数优化。下面是一个简单的参数调整示例：

```r
# 安装并加载必要的包
install.packages("mlr3tuning")
install.packages("paradox")
library(mlr3tuning)
library(paradox)

# 定义搜索空间
search_space = ps(
  gamma = p_dbl(0.001, 0.1),
  cost = p_dbl(1, 100)
)

# 设置随机搜索算法
tuner = tnr("random_search", batch_size = 10)

# 设置网格搜索的终止条件
terminator = trm("evals", n_evals = 50)

# 定义调整实例
instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.acc"),
  search_space = search_space,
  terminator = terminator
)

# 进行参数调整
tuner$optimize(instance)

# 获取最佳参数
best_params = instance$result_learner_param_vals
print(best_params)

# 使用最佳参数重新训练模型
learner$param_set$values = best_params
learner$train(task, row_ids = train_set)

# 在测试集上进行预测
prediction = learner$predict(task, row_ids = test_set)

# 评估预测结果
performance = prediction$score(msr("classif.acc"))
print(performance)
```

通过上述代码，可以调整 `gamma` 和 `cost` 参数，以找到最优的支持向量机模型参数，从而提高分类性能。这展示了如何在 `