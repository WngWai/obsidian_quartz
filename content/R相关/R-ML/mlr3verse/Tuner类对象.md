在R语言的mlr3框架中，mlr3tuning包提供了Tuner类对象，用于执行超参数调优。按照功能分类，下面是Tuner类对象的属性和方法的介绍，并给出了一个应用举例：

**属性**：

1. **param_set**：参数集合，包含了所有要调优的超参数的定义。

2. **learner**：学习器对象，表示要进行超参数调优的学习算法。

3. **store_models**：是否在调优过程中存储每个超参数配置的训练模型。

4. **store_models_control**：定义存储训练模型的方式和选项。

5. **resampling**：重抽样对象，用于评估每个超参数配置的性能。

6. **store_results**：是否在调优过程中存储每个超参数配置的性能结果。

7. **store_results_control**：定义存储性能结果的方式和选项。

8. **termination**：定义调优过程的终止条件。

**方法**：

1. **tune()**：执行超参数调优过程，根据定义的参数范围和搜索策略搜索最佳的超参数配置。

2. **get_result()**：获取调优过程中的最佳超参数配置和性能结果。

3. **get_performance()**：获取给定超参数配置的性能结果。

4. **get_param_set()**：获取参数集合，即所有要调优的超参数的定义。

5. **get_search_space()**：获取搜索空间，即所有要调优的超参数的取值范围。

6. **get_learner()**：获取学习器对象，表示要进行超参数调优的学习算法。

7. **get_resampling()**：获取重抽样对象，用于评估每个超参数配置的性能。

8. **get_tuning_history()**：获取调优过程中的历史结果，包括每个超参数配置和对应的性能。

9. **get_default_tune_control()**：获取默认的调优控制选项。

10. **set_tune_control()**：设置调优控制选项，如搜索策略、迭代次数等。

11. **set_termination()**：设置调优过程的终止条件。

**应用举例**：

下面是一个应用举例，展示如何使用Tuner类对象进行超参数调优：

```R
library(mlr3)
library(mlr3tuning)

# 创建一个任务
task = mlr_tasks$get("iris")

# 创建一个学习器
learner = lrn("classif.rpart")

# 创建一个参数集合
param_set = ParamSet$new(params = list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1)
))

# 创建一个重抽样对象
resampling = rsmp("cv", folds = 5)

# 创建一个Tuner对象
tuner = Tuner$new(learner = learner, param_set = param_set,
                  resampling = resampling)

# 执行超参数调优
tuner$tune(task)

# 获取调优结果
result = tuner$get_result()
best_params = result$learner_param_vals
best_performance = result$performance

print(best_params)
print(best_performance)
```

在这个示例中，我们首先加载了mlr3和mlr3tuning库。然后，创建了一个任务对象`task`，表示要解决的机器学习问题。接下来，创建了一个学习器对象`learner`，使用了rpart算法作为学习算法。

然后，我们创建了一个参数集合`param_set`，其中包含了要调优的超参数的定义。在这个例子中，我们只调优了一个超参数`cp`，它表示rpart算法中的复杂度参数。

接着，我们创建了一个重抽样对象`resampling`，用于评估每个超参数配置的性能。在这个例子中，我们使用了交叉验证方法进行评估。

最后，我们创建了一个Tuner对象`tuner`，指定了学习器、参数集合和重抽样对象。然后，我们调用`tune()`方法执行超参数调优过程，并使用`get_result()`方法获取最佳超参数配置和性能结果。