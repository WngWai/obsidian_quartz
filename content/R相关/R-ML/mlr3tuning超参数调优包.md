在R语言中，`mlr3tuning`是`mlr3`生态系统中的一个包，专门用于调参（参数优化）。这个包提供了一个灵活的框架来**优化机器学习模型的超参数**。通过使用`mlr3tuning`，用户可以定义一个或多个超参数的搜索空间，并应用不同的搜索算法来找到最优的参数组合。

```R
kmeans_lrn = lrn("clust.kmeans")

# 参数空间
search = ps(
  centers = p_int(2,9),
  algorithm = p_fct(levels = c("Hartigan-Wong","Lloyd","MacQueen")),
  nstart = p_int(10,10)
)

# 创建调优实例
instance = TuningInstanceSingleCrit$new(
  task = taskC,
  learner = lrn("clust.kmeans"),
  resampling = rsmp("holdout"),
  measure = msr("clust.dunn"),
  search_space = search,
  terminator = trm("none")
)

# 定义调优器
tuner = tnr("grid_search")

# 进行参数调优
tuner$optimize(instance)

# 调优过程和
instance$archive
autoplot(instance, type = "surface") # 只适用两参数的调优实例

# 查看模型参数调整结果
print(instance$result_learner_param_vals)
print(instance$result)

# （重要）进行学习器参数调整
kmeans_lrn$param_set$values = instance$result_learner_param_vals

## 再根据调整好参数的学习器，训练得到模型！
# 训练模型
kmeans_lrn$train(taskC)
kmeans_lrn$model
# 预测
kmeansPos = kmeans_lrn$predict(taskPos)

# 评估模型得分
kmeansPos$score(msr("classif.acc"))
```

### 参数空间search：
[[paradox 处理参数和超参数]]


### 终止条件terminator:
[[instance=TuningInstanceSingleCrit$new()]] 在调优实例中查看具体定义

## 调优方式一
### 调优实例（Tuning Instance）
这些实例定义了需要**调优的任务、学习器、搜索空间和性能度量**。

[[instance=TuningInstanceSingleCrit$new()]]创建一个**单一标准**的调优实例。

[[TuningInstanceMultiCrit()]]创建一个**多标准**的调优实例。

### 定义调优器
- Tuner$new()

	[[Tuner类对象]]

- tnr()
	
	调优器定义了**搜索超参数的算法**。直接引用调优器
	tuner <- trn("grid_search") **随机搜索**调优
	tuner <- tnr("random_search") **网格搜索**调优，地毯式搜索
	tuner <- tnr("irace") **迭代竞赛算法** (irace) 进行调优
	tuner <- tnr("genetic_search") 基于**遗传算法**的搜索



### 由调优器对调优实例进行调优
tuner\$optimize(instance) 在instance=TuningInstanceSingleCrit\$new()有详细讲解调优和获得结果

~~tuner\$tune(instance)~~没有\$tune()方法！应该是旧包中的调优方法！

### 获得结果
在instance=TuningInstanceSingleCrit\$new()有详细讲解调优和获得结果

instance$archive**执行调优后**，**访问调优历史**

**instance$result**获取调优的**最终结果**。

**instance$result_learner_param_vals**，更为精简，看参数
results <- instance$result
best_params <- results$param_set


## 调优方式二
instance=[tune()](https://mlr3tuning.mlr-org.com/reference/tune.html) 

```R
# Hyperparameter optimization on the Palmer Penguins data set
task = tsk("pima")

# Load learner and set search space
learner = lrn("classif.rpart",
  cp = to_tune(1e-04, 1e-1, logscale = TRUE)
)

# Run tuning
instance = tune(
  tuner = tnr("random_search", batch_size = 2),
  task = tsk("pima"),
  learner = learner,
  resampling = rsmp ("holdout"),
  measures = msr("classif.ce"),
  terminator = trm("evals", n_evals = 4)
)

# Set optimal hyperparameter configuration to learner
learner$param_set$values = instance$result_learner_param_vals

# Train the learner on the full data set
learner$train(task)

# Inspect all evaluated configurations
as.data.table(instance$archive)

```




## 自动化调优？？？
很有些混乱！后面再整理

auto_tuner() 创建AutoTunner对象
[[AutoTuner]]自动调整超参数，是一个更大范围的包含？
