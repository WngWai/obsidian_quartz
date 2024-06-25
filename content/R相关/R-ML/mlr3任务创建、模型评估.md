`mlr3` 是 R 语言中一个强大的机器学习框架，旨在提供一个统一的界面来处理各种机器学习问题。它通过一系列的对象和方法来简化机器学习**任务的创建、模型的训练与评估过程**。

![[Pasted image 20240226115052.png]]

[mlr3官方文档](https://mlr3.mlr-org.com/reference/index.html)

### 任务创建（Tasks）
任务可以理解为**封装后的数据**，模块化操作，方便在学习器上重复利用。

```R
library(mlr3)

# 创建一个任务
task = TaskClassif$new(id = "my_task", backend = iris, target = "Species")

# 创建一个学习器
learner = lrn("classif.rpart")

# 训练模型
learner$train(task)

# 进行预测
predictions = learner$predict(task)

# 查看预测结果
print(predictions)
```

[[data()]]加载指定的数据集到R的工作环境中。as.data.table()将相应数据集转换为data.table结构

- **Task$new()**：创建一个新的任务。

	[[Task任务类对象]]

	- 监督学习和非监督学习

		[[TaskRegr]] 创建**回归任务**，针对定量数据的回归算法
	
		[[TaskClassif]] 创建**分类任务**，针对分类数据的分类算法
	
		[[TaskClust]] **非监督学习任务**，由mlr3cluster包提供

	- 其他
	
		[[TaskSurv]] 包含有时间信息的生存分析算法，该方法在mlr3proba包中

		[[TaskDens]] 非监督学习算法，估计密度，由mlr3proba包提供
	
		TaskRegrST 针对时空数据的回归算法，由mlr3spatiotempcv包提供
		
		TaskOrdinal 等级回归算法，由mlr3ordinal包提供，但是这个包目前正处于开发中，还无法使用


- [[tsk()]]访问和加载mlr3中已经定义的任务

### 评估和比较模型（Resampling and Benchmarks）

```R
library(mlr3)
library(mlr3benchmark)
library(mlr3learners)

# 创建一个任务
task = mlr_tasks$get("iris")

# 创建一个学习器
learner = lrn("classif.rpart")

# 创建一个重抽样对象
resampling = rsmp("cv", folds = 5)

# 创建一个Benchmark对象
benchmark = benchmark(task = task, learners = learner, resampling = resampling)

# 执行Benchmark评估
bmr = benchmark$evaluate()

# 获取Benchmark结果
bmr_results = bmr$aggregate()

print(bmr_results)



**公共方法**：
`$instantiate()` 实例化重采样
`$resample()` 执行重采样过程。
```

![[a60a9d462f5ebe71970085c0d080d1e.png|400]]

- **重抽样对象**
 
	*定义了重抽样过程的**方法和参数***，是抽样的策略！说是重抽样，实则在训练模型时也会用到，在训练模型后再进行重抽样，进行模型评价！
	
	- **Resampling$new()**：创建一个新的重采样策略。

		[[Resampling类对象]] 定义**重采样策略**

	- rsmp()一个快捷函数，用于快速访问**预定义**的重采样策略

		[[holdout]] **留出法**。使用rsmp("holdout")创建一个留出法的重采样实例。

		"cv" 使用rsmp("cv")创建一个**交叉验证的重采样实例**，用于评估模型的泛化能力。（交叉验证将数据集划分为K个子集（通常称为折叠或fold），每次使用其中的K-1个子集作为训练集，剩下的1个子集作为验证集。这个过程会重复K次，每次使用不同的验证集，最终将K次的评估结果进行平均。）
		
		"subsampling"：**子采样**。使用rsmp("subsampling")创建一个子采样的重采样实例。
		
		"repeated_cv"：**重复交叉验证**。使用rsmp("repeated_cv")创建一个重复交叉验证的重采样实例。
		
		"bootstrapping"：**自助法**。使用rsmp("bootstrapping")创建一个自助法的重采样实例。
		
		"time_series"：**时间序列法**。使用rsmp("time_series")创建一个时间序列法的重采样实例

---

- 模型评估
	
	- 单个模型 (1\*1\*1)
	
		[[R相关/R-ML/mlr3verse/resample()|resample()]] **执行重抽样**，对给定的任务和学习器**进行模型训练和性能评估**。通过aggregate()方法获得评估结果
	
		[[ResampleResult对象]] **执行重抽样的结果**

	- 多个模型 (多\*多\*多)
	
		[[benchmark()]]**执行机器学习方法的性能评估和比较**，用于比较**不同任务、不同重抽样、不同学习器（模型算法）的性能**，可以同时对多个任务、学习器和重抽样方法进行评估。

---

手动设置？

three-fold CV
cv3 = rsmp("cv", folds = 3)
cv3$instantiate(tsk_penguins)

first 5 observations in first training set
cv3$train_set(1)[1:5]
[1] 1 9 21 22 23

first 5 observations in third test set
cv3$test_set(3)[1:5]
[1] 2 3 5 10 12


查看详细的内容！？
rr = resample(tsk_penguins, lrn_rpart, cv3)
acc = rr$score(msr("classif.ce"))
rr$aggregate(msr("classif.ce"))

### 重抽样的数据，如何逐步用于训练、建模、评分
但实际，重抽样一般直接用作模型评估了，直接打包了，如resample()，可以结合[[mlr3tuning超参数调优包]]

可行！！！
```R
# 加载数据集
data("car_eval")

# 创建任务
task = TaskClassif$new(id = "car_eval", backend = car_eval, target = "class")

# 使用分层交叉验证
split = rsmp("cv", folds = 5, stratify = TRUE)

# 实例化重采样策略，得到训练集和测试集
split$instantiate(task)

# 获取训练集和测试集索引
train_set = split$train_set(1)
test_set = split$test_set(1)

# 创建 MLP 学习器并配置参数
learner = lrn("classif.keras_mlp",
              epochs = 50,          # 训练轮数
              batch_size = 32,      # 批量大小
              hidden = c(128, 64),  # 隐藏层及其神经元数量
              activation = "relu",  # 激活函数
              optimizer = "adam",   # 优化器
              scale = TRUE)         # 是否归一化

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


### 能不能将模型可视化展示出来？？？
```R
# 可视化结果
# 提取其中一个训练好的模型（第一个模型）
trained_model = rr_rf$learners[[1]]

# 提取其中一棵决策树
tree = trained_model$forest$tree

# 使用mlr3viz包进行可视化
library(rpart.plot)
rpart.plot::rpart.plot(trained_model)
```