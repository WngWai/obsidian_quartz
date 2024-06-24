`mlr3tuning` 包中的 `AutoTuner` 类是一个自动调参的工具，用于优化机器学习模型的超参数。下面按功能分类介绍 `AutoTuner` 类对象的常用属性和方法：

[AutoTuner官网](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html#details)

### 1. 构建和初始化

`AutoTuner` 对象的创建通常是为了自动化超参数的调优过程。在构建时，你需要指定多个关键组件，包括用于调优的学习器、参数搜索空间、调优方法、以及评估标准。

- **构造函数 `AutoTuner$new()`**：创建一个 `AutoTuner` 对象。你需要指定学习器（learner）、调参域（search_space）、调优策略（tuner），以及评价函数（resampling）和评价指标（measure）。

### 2. 属性访问

`AutoTuner` 对象提供了一些属性，可以用来**访问调优过程中的关键信息**。

- **`$learner`**：访问用于调优的学习器。
- **`$resampling`**：获取用于模型评估的重采样策略。
- **`$measure`**：访问用于评估模型性能的度量指标。
- **`$search_space`**：查看定义的参数搜索空间。
- **`$tuner`**：获取用于调优的策略对象。
- **`$tuning_instance`**：访问完成调优后的调优实例，可以从中获取调优结果和详细信息。

### 3. 方法调用

`AutoTuner` 提供了一些方法，使得用户能够执行调优过程、访问调优结果，以及利用最佳参数配置重新训练模型。

- **`$train()`**：在指定的任务上训练 AutoTuner 对象。这个过程包括搜索最优的超参数配置。
- **`$predict()`**：使用经过调优的模型对新数据进行预测。
- **`$score()`**：根据指定的性能度量对模型的预测进行评分。
- **`$clone()`**：克隆一个 AutoTuner 对象，可以用于在不同数据集或设置上重复调优过程。

### 4. 调优结果访问

调优完成后，`AutoTuner` 提供了方法和属性来访问和分析调优结果。

- **`$archive`**：访问调优过程的存档，包含每一步调优的详细结果。
- **`$result`**：获取调优过程的最终结果，通常包含最优的超参数配置。

通过以上的属性和方法，`mlr3tuning` 的 `AutoTuner` 类为 R 语言中的机器学习模型调参提供了一个强大且灵活的自动化工具。



```R
library(mlr3)
library(mlr3tuning)
library(mlr3learners)

# 选择数据集并创建任务
task_penguins <- tsk("penguins")

# 定义学习器和调参空间
learner_rf <- lrn("classif.ranger", predict_type = "prob")
# 定义超参数搜索空间
search_space <- ps(
  mtry = p_int(lower = 2, upper = 8),
  min.node.size = p_int(lower = 1, upper = 5)
)

# 设置调优策略和评价指标
tuner <- tnr("random_search", batch_size = 10)
measure <- msr("classif.acc")

# 创建 AutoTuner 对象并训练
auto_tuner <- AutoTuner$new(
  learner = learner_rf,
  resampling = rsmp("holdout"),
  measure = measure,
  search_space = search_space,
  terminator = trm("evals", n_evals = 20),
  tuner = tuner
)
# 训练 AutoTuner
auto_tuner$train(task_penguins)


# 获取最优参数
print(auto_tuner$result)
# 预测性能
pred <- auto_tuner$predict(task_penguins)
performance <- pred$score(msr("classif.acc"))
print(performance)


```




假设你正在使用`mlr3`包进行机器学习任务，并且你想要优化一个回归模型的超参数。你可以按照以下步骤使用AutoTuner：

1. **定义超参数空间**：首先，你需要定义要优化的超参数空间。例如，你可以定义一个包含学习率（learning rate）、正则化参数（regularization parameter）和决策树的最大深度（max depth）等超参数的范围。

2. **选择评估指标**：确定一个评估指标来衡量模型的性能。例如，你可以选择均方根误差（RMSE）作为评估指标。

3. **创建AutoTuner对象**：使用`mlr3tuning`包中的`AutoTuner`函数，创建一个AutoTuner对象。将回归任务、超参数空间和评估指标作为参数传递给该函数。

4. **运行AutoTuner**：使用`AutoTuner`对象的`tune()`方法，运行AutoTuner。你可以指定要运行的优化次数和其他参数。

5. **获取最佳超参数配置**：一旦AutoTuner完成优化过程，你可以使用`AutoTuner`对象的`get_tuned_hyperpars()`方法获取最佳的超参数配置。


```R
library(mlr3)
library(mlr3tuning)

# 定义回归任务
task = mlr_tasks$get("iris")
learner = mlr_learners$get("regr.rpart")

# 定义超参数空间
param_set = ParamSet$new(
  params = list(
    ParamDbl$new("cp", lower = 0.001, upper = 0.1),
    ParamInt$new("maxdepth", lower = 1, upper = 10)
  )
)

# 创建AutoTuner对象
tuner = AutoTuner$new(
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measure = msr("regr.rmse"),
  search_space = param_set
)

# 运行AutoTuner
tuner$tune(instantiate(task))

# 获取最佳超参数配置
best_hyperpars = tuner$get_tuned_hyperpars()
```

在这个例子中，我们使用了`mlr3`包中的`iris`数据集和回归模型rpart。我们定义了一个包含`cp`（剪枝参数）和`maxdepth`（决策树的最大深度）的超参数空间。然后，我们创建了一个AutoTuner对象，指定了交叉验证（3折）作为评估方法，并使用RMSE作为性能指标。最后，我们运行了AutoTuner，并获取了最佳超参数配置。


###  auto_tuner()和AutoTuner$new()的区别
在`mlr3`这个R语言的机器学习框架中，`auto_tuner()`函数和`AutoTuner$new()`方法提供了一个创建自动调参对象的功能。



1. **`AutoTuner$new()`方法**：这是面向对象编程中创建`AutoTuner`对象的直接方式。`AutoTuner`是mlr3tuning包的一部分，用于自动调整机器学习模型的超参数。使用这个方法时，你需要明确指定所有需要的参数，如学习器（learner），调参方法（tuner），以及评估策略（resampling）等。这种方法给予了用户更细致的控制能力，但同时也要求用户对mlr3的框架有较深的了解。

```r
library(mlr3)
library(mlr3tuning)
# 创建一个 AutoTuner 对象的示例
at <- AutoTuner$new(
  learner = lrn("classif.rpart", cp = to_tune(1e-04, 1e-1, logscale = TRUE)),
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = ps(cp = p_dbl(lower = 1e-04, upper = 1e-1)),
  terminator = trm("evals", n_evals = 100),
  tuner = tnr("grid_search", resolution = 10)
)
```

2. **`auto_tuner()`函数**：这是一个更高层次的辅助函数，旨在简化自动调参对象的创建过程。它内部实际上调用了`AutoTuner$new()`，但提供了一个**更简洁**的界面，自动处理了部分参数的配置。使用`auto_tuner()`时，用户只需要指定最关键的参数，这使得对于不太熟悉mlr3包细节的用户来说，更容易上手。

```r
# 使用 auto_tuner 函数的示例
at2 <- auto_tuner(
  method = "grid_search",
  learner = lrn("classif.rpart"),
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = ps(cp = p_dbl(lower = 1e-04, upper = 1e-1)),
  terminator = trm("evals", n_evals = 100)
)
```

**总结**：`auto_tuner()`函数提供了一个更易用的接口来创建自动调参的对象，适合快速上手和简化代码，而`AutoTuner$new()`方法则提供了更高级的自定义性和控制能力。实际应用中，可以根据需要选择使用哪一种。