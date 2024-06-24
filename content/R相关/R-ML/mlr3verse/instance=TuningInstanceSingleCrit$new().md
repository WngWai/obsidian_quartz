在这个生态系统中，`TuningInstanceSingleCrit`类（或者相似名字，可能会随着版本迭代有所变化）主要用于单一评价标准的调参实例。下面我将基于您提到的`TuningInstanceSingleCrit`（或类似名字）的概念，介绍这类对象的一般属性和重要方法。请注意，具体的函数名称、方法和属性可能会随着`mlr3tuning`包的更新而发生变化。

### 属性
**任务 (task)**: 需要进行调优的机器学习任务，例如分类、回归等。
**学习器 (learner)**: 选定的机器学习算法，即`mlr3`的`Learner`对象。

重抽样（resampling）： rsmp("holdout")，详看[[mlr3任务创建、模型评估]]

**评价标准 (measure)**: 优化的目标，即模型评价的标准，如**精度、AUC**等。
**搜索空间 (search_space)**: 定义了需要优化的超参数及其取值范围。通常使用`ParamSet`对象来定义。
[[paradox 处理参数和超参数]]

**终止条件（terminator）**：terminator = trm("evals", n_evals = 100)

- Terminators$new()
	
	[[Terminators类对象]] 定义调优过程的终止条件

- trm 直接引用内置调优终止条件

	[官网](https://mlr-org.com/terminators.html)

	evals：根据迭代次数终止调优，参数n_evals迭代次数

	trm("none")表示什么没有设终止条件？

**调优历史 (Tuning History)**: 记录了调优过程中所有的评估结果和相应的超参数设置。

### 重要方法
#### 执行调优
tuner$optimize()返回一个**包含每次搜索结果的列表**，优化列表
```R
> tuner$optimize(instance)
INFO  [13:13:02.712] [bbotk] Starting to optimize 3 parameter(s) with '<TunerGridSearch>' and '<TerminatorNone>'

INFO  [13:13:02.765] [bbotk] Evaluating 1 configuration(s)
INFO  [13:13:02.785] [mlr3] Running benchmark with 1 resampling iterations
INFO  [13:13:02.826] [mlr3] Applying learner 'clust.kmeans' on task 'gvhdCtrlScale' (iter 1/1)
INFO  [13:13:02.869] [mlr3] Finished benchmar
INFO  [13:13:04.272] [bbotk] Result of batch 1:
INFO  [13:13:04.282] [bbotk]

INFO  [13:13:04.286] [bbotk] Evaluating 1 configuration(s)
INFO  [13:13:04.297] [mlr3] Running benchmark with 1 resampling iterations
INFO  [13:13:04.304] [mlr3] Applying learner 'clust.kmeans' on task 'gvhdCtrlScale' (iter 1/1)
INFO  [13:13:04.374] [mlr3] Finished benchmark
INFO  [13:13:05.314] [bbotk] Result of batch 2:
INFO  [13:13:05.326] [bbotk]

INFO  [13:13:05.330] [bbotk] Evaluating 1 configuration(s)
INFO  [13:13:05.342] [mlr3] Running benchmark with 1 resampling iterations
INFO  [13:13:05.349] [mlr3] Applying learner 'clust.kmeans' on task 'gvhdCtrlScale' (iter 1/1)
INFO  [13:13:05.377] [mlr3] Finished benchmark
INFO  [13:13:06.109] [bbotk] Result of batch 3:
INFO  [13:13:06.120] [bbotk]
```


#### 调优结果
instance$archive**执行调优后**，**访问调优历史**。这个属性允许用户查看每一步调优的详细结果，包括超参数配置和性能评价。就是上面调优的内容吧？
```R
instance$archive
<ArchiveTuning> with 24 evaluations
```

![[Pasted image 20240427163953.png|400]]


TuningResult(就是result)保存调优结果的对象，包括最佳超参数和相关的性能度量，instance$result查看。
**instance$result**获取调优的**最终结果**。通常包括最佳的超参数设置和对应的性能评价。
![[Pasted image 20240427163803.png|400]]
或者**instance$result_learner_param_vals**，更为精简，看参数
```R
$centers
[1] 5

$algorithm
[1] "Hartigan-Wong"

$nstart
[1] 10
```



5. **构造函数**: 创建调优实例时，需要传递任务、学习器、搜索空间和评价标准等信息。
6. **$plot()**: （如果可用）可视化调优过程和结果，帮助理解调优过程和最终结果。

### 使用示例

以下是一个简化的示例代码，展示了如何使用`mlr3tuning`进行单一标准的调优（注意，真实的使用可能需要根据最新的`mlr3tuning`包版本进行调整）：

```r
library(mlr3)
library(mlr3tuning)

# 定义任务、学习器和评价标准
task <- tsk("iris")
learner <- lrn("classif.rpart")
measure <- msr("classif.ce")

# 定义搜索空间
search_space <- ps(cp = p_dbl(lower = 0.001, upper = 0.1))

# 创建调优实例
tuning_instance <- TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = rsmp("holdout"),
  measure = measure,
  search_space = search_space,
  terminator = trm("evals", n_evals = 100)
)

# 执行调优
tuner <- tnr("random_search")
tuner$optimize(tuning_instance)

# 查看结果
print(tuning_instance$result)
```

请确保查看最新的`mlr3tuning`文档来获取准确的类名和方法，因为软件包随时间更新可能会发生变化。