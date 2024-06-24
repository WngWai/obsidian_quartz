在`mlr3`中，`Learner`是一个非常核心的概念，它封装了各种机器学习算法的实现，包括分类、回归、聚类等。一个`Learner`对象主要包含了模型的训练和预测方法，以及与模型相关的各种参数和属性。

### 主要属性

以下是`Learner`对象的一些主要属性：

- **id**: 学习器的唯一标识符，如`"classif.ranger"`。
- **predict_type**: 预测类型，决定了模型的输出形式，如`"prob"`表示概率输出。
- **feature_types**: 定义学习器可以处理的特征类型。
- **properties**: 描述了学习器的特性，比如是否支持缺失值处理、是否能处理因子型特征等。

- `learn$param_set` **学习器的参数集合**，快速查看参数内容和可以调整的参数及其范围。是params和values的合并效果

learn\$param_set\$params
```R
# 看机器学习器的参数内容
kmeans_lrn=lrn("clust.kmeans")
kmeans_lrn$param_set$params # 跟下面差不多，只是更分散
kmeans_lrn$param_set # 推荐
```
![[Pasted image 20240407104558.png]]

learn\$param_set\$values 学习器**当前实例的实际参数值**，可查意味可改。这些值可能已经在初始化学习器时被设定，或者在之后的某个时刻被用户修改。
```R
# 学习器的参数调整
kmeans_lrn$param_set$values = list(centers = 4, iter.max = 100,nstart = 10)
kmeans_lrn$param_set$values
```
![[Pasted image 20240407162828.png]]

learn\$state
查看学习器内部保存的训练后的模型状态
记经常用到的信息！
```R
kmeans_lrn$train(taskC)
kmeans_lrn$state

# 结果
$model
K-means clustering with 4 clusters of sizes 428, 1488, 4227, 666

Cluster means:
         CD3        CD4          CD8       CD8b
1  1.6843512 -0.9164416  2.796499034  0.5912660
2 -0.4350552 -1.2790155 -0.004331523 -1.2397409
3 -0.3417141  0.2904422 -0.216151841  0.5955009
4  2.0583862  1.6031727 -0.415589260 -1.3896543

Clustering vector:
   [1] 3 3 2 2 2 3 1 3 3 3 3 3 3 2 2 1 2 3 2 2 3 2 3 3 4 3 2 3 3 4 3 3
 [ reached getOption("max.print") -- omitted 5809 entries ]

Within cluster sum of squares by cluster:
[1] 1402.9972 2113.1202 4239.4711  921.7474
 (between_SS / total_SS =  68.1 %)

Available components:

[1] "cluster"      "centers"      "totss"        "withinss"    
[5] "tot.withinss" "betweenss"    "size"         "iter"        
[9] "ifault"      

$log

$train_time
[1] 0.25

$param_vals
$param_vals$centers
[1] 4

$param_vals$iter.max
[1] 100

$param_vals$nstart
[1] 10


$task_hash
[1] "7a7eb859cbe9084a"

$feature_names
[1] "CD3"  "CD4"  "CD8"  "CD8b"

$mlr3_version
[1] ‘0.18.0’

$data_prototype

$task_prototype

$train_task
<TaskClust:gvhdCtrlScale> (6809 x 4)
* Target: -
* Properties: -
* Features (4):
  - dbl (4): CD3, CD4, CD8, CD8b
```


learn\$assignments
查看训练结果？？？
```R
kmeans_lrn$assignments
```


### 主要方法

`Learner`类提供了一系列方法来训练模型和进行预测，以下是一些主要的方法：

- **$train()**: 接受一个`Task`对象作为输入，使用这个任务中的数据来训练模型。
- **$predict()**: 在训练后，使用这个方法对新的数据进行预测。它接受一个`Task`对象，并返回一个`Prediction`对象，其中包含了预测结果。
- **$clone()**: 克隆一个学习器对象，可以用于保持原始学习器的状态，而对克隆出的对象进行修改或训练。

### 示例

以下是创建一个分类学习器，对其训练并进行预测的简单示例：

```R
library(mlr3)
library(mlr3learners)

# 创建一个分类任务
task <- tsk("iris")

# 创建一个ranger分类学习器，设置预测类型为类别概率
learner <- lrn("classif.ranger", predict_type = "prob")

# 训练模型
learner$train(task)

# 创建一个新的Task，用于预测（这里为了简化直接使用了训练集）
new_task <- task

# 进行预测
prediction <- learner$predict(new_task)

# 查看预测结果
print(prediction)
```

在这个示例中，我们首先创建了一个基于`iris`数据集的分类任务。然后，我们初始化了一个`classif.ranger`学习器，设置预测类型为概率输出。接着，我们使用`$train()`方法来训练模型，并用`$predict()`方法进行了预测。最后，我们打印出了预测结果。

`mlr3`提供了一套完整的工具来处理机器学习的整个工作流，从任务创建、模型训练与调优，到预测和评估，都有相应的方法和对象来支持。