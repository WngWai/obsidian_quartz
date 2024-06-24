`mlr3measures` 是一个R语言包，它为机器学习任务提供了一系列的性能评价指标。它是 `mlr3` 生态系统的一部分，`mlr3` 是一个为机器学习任务和流程提供统一框架的R语言包。`mlr3measures` 包中的函数可以按照它们评价的任务类型来分类。

```R
一般都是介绍评估或者超参数调整中使用吧？

# 预测结果
kmeansPos = kmeans_lrn$predict(taskPos)

# 评估模型得分
kmeansPos$score(msr("classif.acc"))

**公共方法**：
`$score()` 计算指标得分。

```


以下是一些常见的 `mlr3measures` 包中的函数，按照它们的功能分类：

- Measures$new() 创建新的评价指标

	[[Measures类对象]]


- **`msr()`** 直接引用已有评估学习器性能的指标，如准确度、AUC等。  

	- 监督学习

		-  回归性能评价指标 (Regression Measures)

			regr.mse均方误差（Mean Squared Error），预测值与真实值差的平方的平均。
			
			regr.rmse均方根误差（Root Mean Squared Error），均方误差的平方根。
			
			`mae`平均绝对误差（Mean Absolute Error），预测值与真实值差的绝对值的平均。
			
			`rsq`决定系数（R-squared），反映模型对数据拟合程度的统计量。

		- 分类性能评价指标 (Classification Measures)

			classif.acc简写为acc。**准确率**（Accuracy），正确分类的样本数除以总样本数。
	
			[[classif.bacc]]: **平衡准确率**（Balanced Accuracy），各个类别的准确率的平均值。**多分类**，并且类之间的**分布差异较大**时
		
			`classif.auc`: **曲线下面积**（Area Under the ROC Curve），通常用于评价二分类模型性能。
			
			logloss**对数损失**（Logarithmic Loss），预测概率的负对数似然。
			
			`f1`: F1分数，精确率和召回率的调和平均数。

	- 非监督学习

		[[mlr3cluster聚类分析]] 详看其中的聚类评价指标


	- 其他

		`mape`: 平均绝对百分比误差（Mean Absolute Percentage Error），预测值和真实值之差的绝对值除以真实值的平均。
		
		`smape`: 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error），类似于MAPE，但是对称化处理，减少了误差的规模依赖性。


### 评价指标的简写
在mlr3中，`msr("classif.acc")`和`msr("acc")`都表示模型性能评估指标"准确率"（Accuracy）。它们在功能上是相同的，都用于计算分类任务的准确率。

区别在于它们的命名约定。`msr("classif.acc")`使用了完整的名称，其中"classif"表示分类任务，"acc"表示准确率。这种命名约定更加明确和具体，可以更好地区分不同类型的评估指标。

而`msr("acc")`是一种简化的命名形式，省略了"classif"部分。在某些情况下，当上下文明确指定了分类任务时，可以使用这种简化的形式。但是，在一些需要明确指定任务类型的情况下，**建议使用完整的命名形式**`msr("classif.acc")`来确保准确性和一致性。


## 逐步求得模型得分和resample()的区别
`resample(task, learner, resampling)` 和 逐步手动训练、预测、评分的过程在本质上有一些关键区别。以下是详细的解释和对比：

```R
resample(task, learner, resampling)和

逐步求得模型得分
task
learner
model = learner$train(task)
prediction = model$predict(task_test)
prediction$score(msr())
```


### `resample(task, learner, resampling)`

#### 特点：
1. **自动化流程**：`resample` 函数封装了训练、验证和评分的整个过程。它会根据给定的重抽样策略（如交叉验证）自动分割数据、训练模型、进行预测和计算评分。
2. **重抽样策略**：可以使用各种重抽样方法，例如交叉验证 (`cv`)、自助法 (`bootstrap`) 等。这些方法用于评估模型在不同数据拆分上的表现。
3. **评估多样性**：通过多次重抽样和评分，可以获得模型性能的稳定估计。例如，在交叉验证中，会对每个折进行训练和验证，从而提供性能的均值和方差。
4. **简洁性**：使用 `resample` 函数可以简化代码，不需要手动编写每一步骤的训练和评估过程。

#### 示例代码：
```r
# 创建分类任务
task = TaskClassif$new(id = "car", backend = cardata_table, target = "class")

# 定义重抽样对象
resampling = rsmp("cv", folds = 5)

# 定义学习器
learner = lrn("classif.rpart")

# 进行重抽样评估
rr = resample(task, learner, resampling)

# 查看评分
scores = rr$score(msr("classif.bacc"))
mean_score = rr$aggregate(msr("classif.bacc"))
```

### 逐步求得模型得分

#### 特点：
1. **手动流程**：需要手动执行训练、预测和评分的每个步骤。可以对每一步进行更精细的控制和调试。
2. **单次评估**：通常用于单次训练和评估模型。在这种情况下，模型只在一个训练集上训练，并在一个测试集上评估。
3. **灵活性**：可以随意调整数据拆分、处理方式和每一步的参数。适合需要复杂数据处理和自定义流程的情况。
4. **步骤清晰**：明确展示了数据从训练到评估的整个过程，有利于理解模型的工作流程。

#### 示例代码：
```r
# 创建分类任务
task = TaskClassif$new(id = "car", backend = cardata_table, target = "class")

# 定义学习器
learner = lrn("classif.rpart")

# 手动分割数据集
split = partition(task, ratio = 0.7)
train_set = split$train
test_set = split$test

# 训练模型
model = learner$train(task, row_ids = train_set)

# 进行预测
prediction = model$predict(task, row_ids = test_set)

# 计算评分
score = prediction$score(msr("classif.bacc"))
```

### 本质区别

1. **自动化 vs 手动**：
   - `resample` 提供了一个高层次的接口来自动化整个重抽样评估过程。
   - 手动训练、预测和评分过程需要用户逐步执行和管理各个步骤。

2. **重抽样策略**：
   - `resample` 可以使用多种重抽样策略，自动化多次数据拆分和评估，适合模型性能的稳定估计。
   - 手动过程通常只进行单次数据拆分和评估，适合特定任务的快速测试和调试。

3. **代码简洁性**：
   - `resample` 函数简洁明了，适合快速评估模型性能。
   - 手动过程代码较为详细，适合需要定制化的评估流程。

4. **结果稳定性**：
   - `resample` 通过多次重抽样提供的性能评估结果更为稳定和可靠。
   - 手动过程的单次评估结果可能受数据拆分的影响较大，不够稳定。

### 选择建议

- **使用 `resample`**：当需要对模型进行标准化评估，并希望自动化整个评估流程时，`resample` 是最佳选择。
- **使用手动流程**：当需要对数据处理和模型训练过程进行详细控制和调试时，手动流程更为合适。

