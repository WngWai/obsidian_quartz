`mlr3viz` 包中的 `autoplot()` 函数是一个泛型函数，旨在提供一种统一的接口来可视化不同的 `mlr3` 对象。根据传入的对象类型不同，它会自动选择适合的可视化方式。`autoplot()` 函数支持多种 `mlr3` 对象，包括任务（Task）、预测（Prediction）、评估结果（ResampleResult、BenchmarkResult）、调优结果（TuningInstance）、特征筛选结果等。

根据输入内容**自动调用对应模型的绘图函数**！

```r
autoplot(object, ...)
```

- `object`: 要可视化的对象。可以是各种 `mlr3` 对象，如 `Task`, `Prediction`, `ResampleResult`, `BenchmarkResult`, `TuningInstance`, 等。

- `type = "reg"`：表示线性回归。
- `type = "pca"`：表示主成分分析。
- `type = "scatter"`：表示散点图。
- `type = "corr"`：表示相关性。

frame = T ???

```r
data(GvHD,package ="mclust")
gvhdCtrlScale <- as.data.table(scale(GvHD.control))
taskC = TaskClust$new("gvhdCtrlScale",gvhdCtrlScale)
class(taskC)
autoplot(taskC)
mlr_learners$keys("clust")
kmeans_lrn=lrn("clust.kmeans")
class(kmeans_lrn)
kmeans_lrn$param_set
kmeans_lrn$param_set$values = list(centers = 4, iter.max = 100,nstart = 10)
kmeans_lrn$param_set

kmeans_lrn$train(taskC)
kmeans_lrn$state
kmeans_lrn$assignments

# 查看绘图
kmeans_lrn1 = kmeans_lrn$clone()
kmeans_lrn1$predict(taskC)
autoplot(kmeans_lrn1$predict(taskC), taskC, type ="scatter" )
```



[autoplot()官网](https://mlr-org.com/gallery/technical/2022-12-22-mlr3viz/index.html#glmnet) 直接参考官网

### Task
#### Classification


#### Regression


#### Cluster



### Instance
调优实例

- 两参数
	autoplot(instance, type = "surface")绘制表面图（surface plot），二维平面上的三维视觉表现

- 多参数
	autoplot(instance, type = "parallel")平行坐标图
	autoplot(instance, type = "pairs")散点图矩阵

autoplot(instance, type = "parameter")  绘制每个参数的效果
autoplot(instance, type = "performance")  绘制每个参数设置的性能


### Learn



### Prediction


