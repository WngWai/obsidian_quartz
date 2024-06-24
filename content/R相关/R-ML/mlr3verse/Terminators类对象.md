在R语言的mlr3包中，Terminators类对象用于**定义调优过程的终止条件**。下面按照功能分类介绍Terminators类对象的属性和方法，并给出一个应用举例：

**属性**：

1. **term_evals**：指定调优过程允许的最大评估次数。

2. **term_time**：指定调优过程允许的最大时间。

3. **term_perf**：指定调优过程达到的最佳性能。

**方法**：

1. **term()**：检查是否满足终止条件。

2. **reset()**：重置终止条件。

**应用举例**：

下面是一个应用举例，展示如何使用Terminators类对象定义调优过程的终止条件：

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

# 创建一个Terminators对象
terminator = Terminators$new(
  term_evals = 100,  # 最大评估次数为100
  term_time = 60,  # 最大时间为60秒
  term_perf = 0.95  # 最佳性能为0.95
)

# 设置Tuner的终止条件
tuner$set_termination(terminator)

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

然后，我们创建了一个Tuner对象`tuner`，指定了学习器、参数集合和重抽样对象。接着，我们创建了一个Terminators对象`terminator`，设置了终止条件，包括最大评估次数为100次，最大时间为60秒，最佳性能为0.95。

最后，我们使用`tuner$set_termination()`方法将Terminators对象应用到Tuner对象上，然后调用`tune()`方法执行超参数调优过程，并使用`get_result()`方法获取最佳超参数配置和性能结果。

这个示例展示了如何使用Terminators类对象在mlr3中定义调优过程的终止条件。通过设置最大评估次数、最大时间或最佳性能，可以灵活控制调优过程的终止条件。具体的调优控制选项和更复杂的应用场景可以参考mlr3和mlr3tuning的官方文档和示例。