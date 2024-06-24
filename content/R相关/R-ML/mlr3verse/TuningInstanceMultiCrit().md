在R语言的mlr3包中，TuningInstanceMultiCrit()类对象用于多目标超参数调优。下面按照功能分类介绍TuningInstanceMultiCrit()类对象的常用属性和方法，并给出一个应用举例：

**属性**：

1. **task**：任务对象，表示要解决的机器学习问题。

2. **learner**：学习器对象，表示要进行超参数调优的学习算法。

3. **resampling**：重抽样对象，用于评估每个超参数配置的性能。

4. **store_models**：是否在调优过程中存储每个超参数配置的训练模型。

5. **store_models_control**：定义存储训练模型的方式和选项。

**方法**：

1. **optimize()**：执行多目标超参数调优过程，根据定义的优化目标搜索最佳的超参数配置。

2. **get_result()**：获取调优过程中的最佳超参数配置和性能结果。

3. **get_performance()**：获取给定超参数配置的性能结果。

4. **get_param_set()**：获取参数集合，即所有要调优的超参数的定义。

5. **get_learner()**：获取学习器对象，表示要进行超参数调优的学习算法。

6. **get_resampling()**：获取重抽样对象，用于评估每个超参数配置的性能。

7. **get_tuning_history()**：获取调优过程中的历史结果，包括每个超参数配置和对应的性能。

8. **get_default_tune_control()**：获取默认的调优控制选项。

9. **set_tune_control()**：设置调优控制选项，如搜索策略、迭代次数等。

**应用举例**：

下面是一个应用举例，展示如何使用TuningInstanceMultiCrit()类对象进行多目标超参数调优：

```R
library(mlr3)
library(mlr3tuning)

# 创建一个任务
task = mlr_tasks$get("iris")

# 创建一个学习器
learner = lrn("classif.rpart")

# 创建一个重抽样对象
resampling = rsmp("cv", folds = 5)

# 创建一个TuningInstanceMultiCrit对象
tuner = TuningInstanceMultiCrit$new(
  task = task,
  learner = learner,
  resampling = resampling,
  store_models = TRUE
)

# 执行多目标超参数调优
tuner$optimize()

# 获取调优结果
result = tuner$get_result()
best_params = result$learner_param_vals
best_performance = result$performance

print(best_params)
print(best_performance)
```

在这个示例中，我们首先加载了mlr3和mlr3tuning库。然后，创建了一个任务对象`task`，表示要解决的机器学习问题。接下来，创建了一个学习器对象`learner`，使用了rpart算法作为学习算法。

然后，我们创建了一个重抽样对象`resampling`，用于评估每个超参数配置的性能。在这个例子中，我们使用了交叉验证方法进行评估。

接着，我们创建了一个TuningInstanceMultiCrit对象`tuner`，指定了任务、学习器和重抽样对象。我们还设置了`store_models`属性为TRUE，表示在调优过程中存储每个超参数配置的训练模型。

最后，我们调用`tuner$optimize()`方法执行多目标超参数调优过程，并使用`get_result()`方法获取最佳超参数配置和性能结果。

这个示例展示了如何使用TuningInstanceMultiCrit类对象在mlr3中进行多目标超参数调优。通过设置任务、学习器和重抽样对象，以及调用相应的方法，可以执行多目标优化并获取最佳的超参数配置和性能结果。具体的调优控制选项和更复杂的应用场景可以参考mlr3和mlr3tuning的官方文档和示例。