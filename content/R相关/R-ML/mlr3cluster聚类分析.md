`mlr3cluster`是专门为聚类任务设计的 R 包，它是`mlr3`生态系统的一部分，专门用于**聚类分析**。在`mlr3cluster`中，聚类任务被视为无监督学习的问题，它提供了统一的接口来访问和评估各种聚类算法。
```r
library(mlr3)
library(mlr3cluster)

# 创建聚类任务
task <- TaskClust$new(id = "my_clust", backend = iris, target = "Species")

# 创建学习器：这里使用 K-means 算法
learner <- lrn("clust.kmeans", centers = 3)

# 训练模型
learner$train(task)

# 预测
predictions <- learner$predict(task)

# 查看预测结果
print(predictions)
```

[mlr3cluster聚类学习器官网](https://mlr3cluster.mlr-org.com/)
### 聚类学习器
- LearnerClust$new() 创建新的聚类学习器
	
	[[LearnerClust类对象]] 


- **`lrn()`**: 此函数用于创建一个特定的聚类学习器（例如 k-means, DBSCAN）。

	- 基于距离的方法
	
		clust.kmeans ：**K-means**最广泛使用的聚类算法之一，通过迭代选择聚类中心，以最小化每个点到其最近的聚类中心的距离平方和。

		**K-medoids**（PAM, Partitioning Around Medoids）：与 K-means 类似，但聚类中心必须是数据点，使得它对噪声和异常值更鲁棒。

	- 基于层次的方法（分层聚类）
	
		虽然分层聚类中距离可以固定下来，但两种分层聚类还是不同的！
		
		clust.agnes 聚集式，更偏向底层结构。**凝聚层次聚类**（Agglomerative Hierarchical Clustering）：从每个点作为单独的簇开始，重复合并最相近的簇，直到满足某个终止条件。
		
		clust.diana 分裂式，更偏向顶层结构。**分裂层次聚类**（Divisive Hierarchical Clustering）：从所有点作为一个簇开始，重复分裂至每个点为一个簇或满足某个终止条件。

	-  基于密度的方法

		clust.dbscan（Density-Based Spatial Clustering of Applications with Noise）：**DBSCAN**，通过将密集连接的区域划分为簇，可以发现任意形状的簇，并对噪声有很好的鲁棒性。

		**OPTICS**（Ordering Points To Identify the Clustering Structure）：与 DBSCAN 类似，但改进了对不同密度区域的聚类能力。


## 聚类评价指标
不在[[mlr3measures模型性能评价包]]包中！

直接msr()调用。

| ID                                                                                               | Measure                              | Package                                               |
| :----------------------------------------------------------------------------------------------- | :----------------------------------- | :---------------------------------------------------- |
| [clust.dunn](https://mlr3cluster.mlr-org.com/reference/mlr_measures_clust.dunn.html)             | Dunn index                           | [fpc](https://cran.r-project.org/package=fpc)         |
| [clust.ch](https://mlr3cluster.mlr-org.com/reference/mlr_measures_clust.ch.html)                 | Calinski Harabasz Pseudo F-Statistic | [fpc](https://cran.r-project.org/package=fpc)         |
| [clust.silhouette](https://mlr3cluster.mlr-org.com/reference/mlr_measures_clust.silhouette.html) | Rousseeuw’s Silhouette Quality Index | [cluster](https://cran.r-project.org/package=cluster) |
| [clust.wss](https://mlr3cluster.mlr-org.com/reference/mlr_measures_clust.wss.html)               | Within Sum of Squares                | [fpc](https://cran.r-project.org/package=fpc)         |




