在`mlr3learners`包中，每种学习器对应于特定的机器学习算法。在`mlr3`框架中，学习器主要负责模型的训练和预测。以下是按功能分类的一些重要的函数和学习器介绍。
mlr3中学习器就是模型，在学习器里指定具体的算法和设置超参数。

```r
library(mlr3)
library(mlr3learners)

# 创建一个学习器
learner <- lrn("classif.ranger", num.trees = 100, predict_type = "prob")

# 创建任务
task <- TaskClassif$new(id = "my_task", backend = iris, target = "Species")

# 训练模型
learner$train(task)

# 预测
predictions <- learner$predict(task)
```

```python
# 将数据集转换为data.table
Life_tb <- as.data.table(Life_tib)

# 构建聚类任务，数据集执行标准化简化训练复杂度
task = TaskClust$new(id = 'k_mean', backend = scale(Life_tb))
class(task)
autoplot(task) # 任务（实质对封装的数据集）可视化

# 定义聚类学习器
learner = lrn('clust.kmeans')

# 查看学习器实际参数和学习器参数信息
learner$param_set
learner$param_set$params # 查看默认参数值信息
learner$param_set$values # 查看实际参数值

# 模型训练，显式创建存储训练模型的变量
model = learner$train(task)

# 模型状态
model$state
# 训练结果
model$assignments

# 按照K=2的默认分类，绘制模型训练效果
autoplot(model$predict(task), task, type = "scatter")
autoplot(model$predict(task), task, type = "pca", frame = T)
autoplot(model$predict(task), task, type = "sil", frame = T)
```

### 公共方法
**训练模型**
$train()方法训练模型。

**预测**：
$predict()方法进行预测。

`update()`：用于更新对象属性的方法，例如改变模型参数或修改数据集???

其他详看[[Learner学习器类对象的基类]]，
如
model$state 查看训练的状态
model$assignments 查看训练结果


- `mlr_learners$keys("关键字")`根据**关键字（正则表达式）** 列出所有在mlr3框架的学习器id（或者说名称）。mlr_learners是一个对象，它包含了**所有已经注册的学习器**（也称为算法或模型）。这些学习器用于不同类型的任务，比如回归、分类、聚类等。
	
	mlr_learners$keys("class") 查找分类学习器
	
	mlr_learners$keys("clust")快速找到mlr3中已有的聚类模型


- [[常用学习器汇总]]  对于查找到的学习器，通过R自带的**help查看参数**，记住常用学习器的常用参数作用


- **Learner$new()**：创建一个**新的学习器**。

	[[Learner学习器类对象的基类]]
	
	- **公共方法**：
	
		$clone()**克隆**学习器对象。
		
		$train()学习器的方法，用于在给定的任务上**训练模型**。
		
		$predict()学习器的方法，用于对新数据进行**预测**。

---
[Learners官网](https://mlr3learners.mlr-org.com/)

|ID|学习器|所属包|
|---|---|---|
|[regr.cv_glmnet](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.cv_glmnet.html)|Penalized Linear Regression|[glmnet](https://cran.r-project.org/package=glmnet)|
|[regr.glmnet](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.glmnet.html)|Penalized Linear Regression|[glmnet](https://cran.r-project.org/package=glmnet)|
|[regr.kknn](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.kknn.html)|k-Nearest Neighbors|[kknn](https://cran.r-project.org/package=kknn)|
|[regr.km](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.km.html)|Kriging|[DiceKriging](https://cran.r-project.org/package=DiceKriging)|
|[regr.lm](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.lm.html)|Linear Regression|stats|
|[regr.nnet](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.nnet.html)|Single Layer Neural Network|[nnet](https://cran.r-project.org/package=nnet)|
|[regr.ranger](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.ranger.html)|Random Forest|[ranger](https://cran.r-project.org/package=ranger)|
|[regr.svm](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.svm.html)|Support Vector Machine|[e1071](https://cran.r-project.org/package=e1071)|
|[regr.xgboost](https://mlr3learners.mlr-org.com/reference/mlr_learners_regr.xgboost.html)|Gradient Boosting|xgboost|

| ID                                                                                                      | 学习器                                   | 所属包                                                   |     |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------- | --- |
| [classif.cv_glmnet](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.cv_glmnet.html)     | Penalized Logistic Regression         | [glmnet](https://cran.r-project.org/package=glmnet)   |     |
| [classif.glmnet](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.glmnet.html)           | Penalized Logistic Regression         | [glmnet](https://cran.r-project.org/package=glmnet)   |     |
| [classif.kknn](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.kknn.html)               | k-Nearest Neighbors                   | [kknn](https://cran.r-project.org/package=kknn)       |     |
| [classif.lda](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.lda.html)                 | LDA (Linear Discriminant Analysis)    | [MASS](https://cran.r-project.org/package=MASS)       |     |
| [classif.log_reg](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.log_reg.html)         | Logistic Regression                   | stats                                                 |     |
| [classif.multinom](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.multinom.html)       | Multinomial log-linear model          | [nnet](https://cran.r-project.org/package=nnet)       |     |
| [classif.naive_bayes](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.naive_bayes.html) | Naive Bayes                           | [e1071](https://cran.r-project.org/package=e1071)     |     |
| [classif.nnet](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.nnet.html)               | Single Layer Neural Network           | [nnet](https://cran.r-project.org/package=nnet)       |     |
| [classif.qda](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.qda.html)                 | QDA (Quadratic Discriminant Analysis) | [MASS](https://cran.r-project.org/package=MASS)       |     |
| [classif.ranger](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.ranger.html)           | Random Forest                         | [ranger](https://cran.r-project.org/package=ranger)   |     |
| [classif.svm](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.svm.html)                 | Support Vector Machine                | [e1071](https://cran.r-project.org/package=e1071)     |     |
| [classif.xgboost](https://mlr3learners.mlr-org.com/reference/mlr_learners_classif.xgboost.html)         | Gradient Boosting                     | [xgboost](https://cran.r-project.org/package=xgboost) |     |

- [[lrn()]]：一个快捷函数，用于**快速访问预定义的学习器**。

	-  回归（Regression）

		[[regr.lm]] 使用R基础包lm()中的**线性模型进行回归分析**。
		
		regr.ranger 使用`ranger`包实现的随机森林算法进行回归。

		[[regr.nnet]] 单层神经网络，**MLP多层感知机**，前馈网络，全连接层

		`lrn("regr.glmnet")`: 使用`glmnet`包实现的弹性网正则化线型模型进行回归。




	- 分类（Classification）

		- 二分类

			`classif.log_reg` 逻辑回归

		- 多分类
		
			[[classif.multinom]] softmax回归

		[[class.nnet]] 用于解决**单层神经网络**

		[[classif.svm]]使用`e1071`包实现的**支持向量机（SVM）**进行分类。也可参考[[package-e1071#svm()]]

		[[classif.rpar]] **决策树**分类器，且是CART类型的，基于**基尼系数**，二叉树。rpar.plot()这个绘制决策图还得注意下！！！

		[[classif.C50]]？？ **决策树**分类器，基于C4.5**信息增益率**，多叉树
		
		[[classif.ranger]]使用`ranger`包实现的随机森林算法进行分类，是`ranger`包在R语言中实现的**随机森林算法**的一个接口。

		`lrn("classif.xgboost")`: 使用`xgboost`包实现的**XGBoost算法**进行分类。

		`lrn("classif.glmnet")`: 使用`glmnet`包实现的弹性网正则化线性模型进行分类。

	- 聚类（Clustering）

		虽然`mlr3learners`主要集中在监督学习算法上，但`mlr3`生态系统也支持聚类分析。聚类通常通过`mlr3cluster`包实现，并且可以通过类似的接口使用。

		[[mlr3cluster聚类分析]] 主要用于定义聚类学习算法，还包括任务和评价指标



### 是否使用model
```python
dbscan_Irn$train(task) 
dbPred = dbscan_lrn$predict(task) 

# 和 

model = dbscan_Irn$train(task) 
dbPred = model$predict(task)
```

- 第一段代码没有显式地创建一个表示训练后模型的变量，而是**隐式**地依赖 `dbscan_lrn` 对象的内部状态。
- 第二段代码**显式**地创建了一个名为 `model` 的变量来存储训练后的模型，使得模型的管理和使用更加清晰。


`mlr3` 是一个强大的 R 包，用于机器学习任务的管理和执行。它提供了一个统一的接口来处理各种机器学习算法，包括决策树和集成算法。以下是如何在 `mlr3` 中实现决策树和集成算法的步骤。

## 待整理，集成算法！
### 实现决策树算法

在 `mlr3` 中，可以使用 `rpart` 包来实现决策树算法。以下是一个简单的示例：

1. **安装和加载必要的包**：
   ```R
   install.packages("mlr3")
   install.packages("mlr3learners")
   install.packages("rpart")
   library(mlr3)
   library(mlr3learners)
   ```

2. **创建任务和学习器**：
   ```R
   # 加载数据集
   data("iris", package = "datasets")

   # 创建分类任务
   task = TaskClassif$new(id = "iris", backend = iris, target = "Species")

   # 创建决策树学习器
   learner = lrn("classif.rpart")
   ```

3. **训练模型**：
   ```R
   # 训练模型
   learner$train(task)
   ```

4. **预测和评估**：
   ```R
   # 进行预测
   prediction = learner$predict(task)

   # 评估模型性能
   prediction$score(msr("classif.acc"))
   ```

### 实现集成算法

在 `mlr3` 中，可以使用 `mlr3learners` 包中的集成算法，如随机森林和梯度提升树。以下是如何实现这些集成算法的示例：

#### 随机森林

1. **安装和加载必要的包**：
   ```R
   install.packages("randomForest")
   library(randomForest)
   ```

2. **创建随机森林学习器**：
   ```R
   # 创建随机森林学习器
   learner_rf = lrn("classif.randomForest")
   ```

3. **训练和评估模型**：
   ```R
   # 训练模型
   learner_rf$train(task)

   # 进行预测
   prediction_rf = learner_rf$predict(task)

   # 评估模型性能
   prediction_rf$score(msr("classif.acc"))
   ```

#### 梯度提升树（使用 xgboost）

1. **安装和加载必要的包**：
   ```R
   install.packages("xgboost")
   library(xgboost)
   ```

2. **创建梯度提升树学习器**：
   ```R
   # 创建梯度提升树学习器
   learner_gbt = lrn("classif.xgboost")
   ```

3. **训练和评估模型**：
   ```R
   # 训练模型
   learner_gbt$train(task)

   # 进行预测
   prediction_gbt = learner_gbt$predict(task)

   # 评估模型性能
   prediction_gbt$score(msr("classif.acc"))
   ```

### 使用 `mlr3pipelines` 进行集成学习

`mlr3pipelines` 包提供了更高级的功能，可以实现更复杂的集成学习方法，如堆叠（Stacking）。

1. **安装和加载必要的包**：
   ```R
   install.packages("mlr3pipelines")
   library(mlr3pipelines)
   ```

2. **创建基础学习器和堆叠学习器**：
   ```R
   # 创建基础学习器
   learner1 = lrn("classif.rpart")
   learner2 = lrn("classif.randomForest")

   # 创建堆叠学习器
   stack = po("learner", learner1) %>>%
           po("learner", learner2) %>>%
           po("featureunion") %>>%
           po("learner", lrn("classif.log_reg"))

   # 创建堆叠任务
   graph_learner = GraphLearner$new(stack)
   ```

3. **训练和评估堆叠模型**：
   ```R
   # 训练堆叠模型
   graph_learner$train(task)

   # 进行预测
   prediction_stack = graph_learner$predict(task)

   # 评估模型性能
   prediction_stack$score(msr("classif.acc"))
   ```

通过这些步骤，你可以在 `mlr3` 中实现决策树和各种集成算法，并进行模型训练和评估。`mlr3` 提供了一个灵活且强大的框架，使得处理机器学习任务变得更加简便和高效。