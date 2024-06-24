
## PCA()

^cfdb22

`FactoMineR`是一个专门用于**多元数据分析的R包**，它提供了各种探索性数据分析方法，如PCA、对应分析（CA）、多元对应分析（MCA）、聚类分析等。

`PCA()`函数在`FactoMineR`包中用于执行主成分分析，它提供了丰富的结果输出和绘图选项。

```python
PCA(X, scale.unit, ncp, graph)
```

- `X`: 要进行PCA的数据矩阵或数据框（data frame）。

- `scale.unit`: 一个逻辑值，指示是否要在分析之前对变量默认进行**标准化**（均值为0，标准差为1）。

scale.unit=TRUE

- `ncp`: 要保留的**主成分的数量**。

- `graph`: 一个逻辑值，指示是否在执行PCA时**自动绘制图形**。

还有其他一些参数可以用来控制PCA的其他方面
`quali.sup`（用于指定分类变量）
`quanti.sup`（用于指定数量补充变量），以及多个用于细化图形输出的参数。


```R
# 首先安装并加载FactoMineR包
if(!require(FactoMineR)){
    install.packages("FactoMineR")
}
library(FactoMineR)

# 运行PCA分析
data(iris)
res.pca <- PCA(iris[, -5], scale.unit = TRUE, ncp = 5)

# 查看PCA结果的摘要
summary(res.pca)

# 绘制变量和个体的图形结果
plot(res.pca, choix = "var") # Variables plot
plot(res.pca, choix = "ind") # Individuals plot
```

在这个例子中，我们首先检查是否已经安装了`FactoMineR`包，如果没有则安装它，并加载该包。然后，我们使用`PCA()`函数在鸢尾花数据集的前四个特征上执行PCA分析，并且请求所有的主成分。我们通过`scale.unit = TRUE`参数来标准化数据。使用`summary()`函数可以查看PCA结果的摘要。最后，我们利用`plot()`函数绘制变量和个体的图形结果，其中`choix`参数用于选择是绘制变量图还是个体图。