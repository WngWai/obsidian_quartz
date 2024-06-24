在R语言中，`factoextra`包是用于**对因子分析和聚类分析结果进行可视化和解释的包**。下面是`factoextra`包中主要函数的介绍，按照功能进行分类。

`factoextra`包提供了一系列便利的函数，用于帮助用户更容易地**提取和可视化多元分析的结果**，包括主成分分析（PCA）、对应分析（CA）、多维尺度分析（MDS）等。

## 单独获得PCA的结果
从PCA对象中获得更精确的PCA结果？？？

### get_pca()

![[Pasted image 20240421142055.png]]

`get_pca()`函数的目的是从**多种PCA对象中提取主要成分的结果**并返回一个**容易处理的列表格式**。这个函数可以处理`prcomp`、`princomp`、`PCA`（来自FactoMineR包）、`dudi.pca`（来自ade4包）等对象。

```R
get_pca(x, element = c("var", "ind"))
```

- `x`: 进行PCA分析的对象，可以是`prcomp`、`princomp`、`PCA`、`dudi.pca`等。
- `...`: 其他参数，一般是传递给特定PCA对象的参数。

```r
# 加载数据（例如：USArrests数据集）
data("USArrests")

# 标准化数据并进行PCA
pca_result <- prcomp(USArrests, scale = TRUE)

# 使用factoextra的get_pca函数提取PCA结果
pca_data <- get_pca(pca_result)
```

现在，我们提取了PCA的结果，并可以对这些结果进行进一步的分析或可视化。例如，我们可以使用`fviz_pca_ind`函数来可视化PCA的个体（观测点）：

```r
# 可视化PCA的个体（观测点）
fviz_pca_ind(pca_data)

fviz_pca_ind(pca_result) # 这样也可以
```

也可以使用`fviz_pca_var`函数来可视化PCA的变量：

```r
# 可视化PCA的变量
fviz_pca_var(pca_data)
```

这些函数`fviz_pca_*`都是`factoextra`包中的函数，它们利用了`ggplot2`包的图形系统来创建易于理解的图形。

### get_pca_ind()


### get_pca_var()


## 因子分析结果可视化

`fviz_screeplot()`：绘制屏幕图，显示各个主成分的方差贡献率。
`fviz_eig()`：绘制特征值图，显示各个主成分的特征值。
`fviz_contrib()`：绘制个体或变量对主成分的贡献度。

fviz_pca_ind()绘制**个体**在主成分空间中的投影图。
fviz_pca_var()绘制**变量**在主成分空间中的投影图。
fviz_pca_biplot()双标图，上面**两者的结合**！

fviz_screeplot()碎石图，根据肘点原则，通过看拐肘所处位置，判定**主成分数量**

### fviz_pca_biplot()双标图
`fviz_pca_biplot()` 函数用于绘制 PCA 分析的**双标图**（biplot），这种图同时展示了观测值（个体）和变量**在主成分空间中的位置**。

![[Pasted image 20240317112117.png]]

```R
fviz_pca_biplot(X, geom = c("point", "text"), axes = c(1, 2), ...)
```

主要参数介绍：

- `X`: PCA 对象，这个对象可以通过 `prcomp()`、`PCA()` (来自 `FactoMineR` 包) 等函数获得。
- `geom`: 图层的几何对象。可以是 `"point"`、`"text"` 或它们的结合。`"point"` 表示用点表示观测值，`"text"` 表示变量以文本的形式显示。
- `axes`: 一个包含两个元素的向量，指定了要绘制的主成分轴，默认是第一和第二主成分。
- `...`: 其他参数可以被传递给底层的绘图函数，例如 `ggplot2` 的图层。


```R
# 加载必要的包
library(factoextra)
library(FactoMineR)

# 对 iris 数据集进行 PCA 分析
data(iris)
res.pca <- PCA(iris[,1:4], graph = FALSE) # 'graph = FALSE' 阻止 PCA() 自动绘图

# 绘制 PCA 双标图
fviz_pca_biplot(res.pca, geom = c("point", "text"), label = "var", 
                col.ind = iris$Species, # 用不同颜色表示不同的物种
                palette = c("#00AFBB", "#E7B800", "#FC4E07"), 
                addEllipses = TRUE) # 为每个组添加置信椭圆
```

在这个例子中，`iris[, 1:4]` 表示使用 `iris` 数据集的前四列（特征）进行 PCA 分析。`geom = c("point", "text")` 表示同时用点和文本表示观测值和变量。`col.ind = iris$Species` 表示用不同的颜色表示 `iris` 数据集的不同物种。最后，`addEllipses = TRUE` 添加了表示各组置信椭圆的图层，以更好地可视化组内的差异。

### fviz_pca_ind()散点图
这个函数是用来可视化主成分分析（PCA）中的观测值（个体）的位置。`fviz_pca_ind()` 会生成一个散点图，用以**展示观测值在选定的主成分轴上的分布**。

散点图，并且标了数字！
![[Pasted image 20240317113413.png]]

```R
fviz_pca_ind(X, axes = c(1, 2), geom = c("point", "text"), ...)
```

主要参数介绍：

- `X`: PCA 对象，可以通过 `prcomp()`、`PCA()` (来自 `FactoMineR` 包) 等函数获得。
- `axes`: 一个包含两个元素的向量，指定了要绘制的主成分轴，默认是第一和第二主成分。
- `geom`: 图层的几何对象类型。可以是 `point`、`text` 或它们的结合。`"point"` 表示用点表示观测值，`"text"` 表示观测值以文本形式显示。
- `...`: 其他参数可以传递给 `ggplot2` 的绘图函数，比如可以指定颜色、大小、形状等。


首先，确保 `factoextra` 包已安装并加载：

```R
install.packages("factoextra")
library(factoextra)
```

随后，使用 R 的内置数据集 `iris` 进行 PCA 分析，并用 `fviz_pca_ind()` 函数来绘制观测值的散点图：

```R
library(factoextra)
library(FactoMineR)

# 对 iris 数据集进行 PCA 分析
data(iris)
res.pca <- PCA(iris[,1:4], graph = FALSE) # 'graph = FALSE' 阻止 PCA() 自动绘图

# 绘制 PCA 观测值的散点图
fviz_pca_ind(res.pca,
             col.ind = iris$Species, # 用不同颜色表示不同的物种
             palette = c("#00AFBB", "#E7B800", "#FC4E07"), 
             addEllipses = TRUE, # 为每个组添加置信椭圆
             legend.title = "Species")
```

在这个例子中，`iris[, 1:4]` 指定了用于 PCA 分析的数据集的列（特征）。`col.ind = iris$Species` 指定了用于分类观测值的变量，这里是根据物种对点进行颜色分类。`addEllipses = TRUE` 用于为每种物种添加置信椭圆以显示其在主成分空间中的分布范围。`legend.title` 用于设置图例标题。

注意，为了生成 PCA 对象，这里使用了 `FactoMineR` 包中的 `PCA()` 函数。 `factoextra` 包可以与 `FactoMineR` 包很好地协作，但也可以接受来自 `stats` 包中 `prcomp()` 或 `princomp()` 函数的 PCA 对象。

### fviz_pca_var()载荷图

![[Pasted image 20240317113243.png]]

`fviz_pca_var()`函数用于可视化PCA分析中的变量。它通过创建一个变量的载荷图，来展示**各个变量如何对主成分的构成做出贡献**。

以下是`fviz_pca_var()`函数的一些重要参数：

- `X`: 一个PCA对象，通常是使用`prcomp()`或`PCA()`函数产生的。
- `axes`: 选择要展示的主成分轴的编号，默认是1和2。
- `col.var`: 设置变量点的颜色。
- `select.var`: 允许基于贡献率或坐标选择部分变量进行展示。
- `repel`: 一个逻辑值，决定是否使用文字避让，使得标签不会相互重叠。
- `...`: 其他参数，如图形参数（如`cex`, `pt.cex`, `pt.size`, `alpha`等）和ggplot2相关的参数。


假设你已经执行了PCA分析，并且想要可视化变量在前两个主成分上的作用，你可以这样做：

```R
# 首先安装并加载factoextra包
if(!require(factoextra)){
    install.packages("factoextra")
}
library(factoextra)

# 运行PCA分析
data(iris)
res.pca <- prcomp(iris[, -5], scale = TRUE)

# 可视化变量（载荷图）
fviz_pca_var(res.pca, col.var = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE)
```

在这个例子中，我们首先检查是否已经安装了`factoextra`包，如果没有则安装它，并加载该包。之后，我们使用`prcomp()`函数在鸢尾花数据集上执行PCA分析。然后，使用`fviz_pca_var()`函数来创建一个变量的散点图，其中变量的颜色基于它们对主成分的贡献。我们使用`repel = TRUE`来避免标签重叠。

请注意，`factoextra`和它的函数`fviz_pca_var()`可能会随着包的更新而改变，所以最佳的做法是查看最新的包文档和函数帮助页面，了解最新的参数和用法。

### fviz_screeplot()碎石图
scree \[skri\] 碎石。指定PCA()中分类数量对碎石土好像没有影响？
`fviz_screeplot()` 函数也属于 R 语言的 `factoextra` 包。这个函数主要用于绘制主成分分析（PCA）或因子分析中的**碎石图**（scree plot），帮助用户判断应该**保留多少个主成分**。

![[Pasted image 20240317112149.png]]

```R
fviz_screeplot(object, ncp = 5, addlabels = FALSE, ...)
```

主要参数介绍：

- `object`: PCA 或因子分析的结果对象。通常是 `PCA()`（来自 `FactoMineR` 包），`prcomp()` 或 `princomp()`（来自 `stats` 包）的结果。
- `ncp`: 要显示的主成分数量，默认为5。
- `addlabels`: 是否在图上添加每个点的标签。默认是 FALSE。
- `...`: 其他参数可以传递给 `ggplot2` 的绘图函数，比如可以指定颜色等。

在使用 `fviz_screeplot()` 函数之前，你需要确保 `factoextra` 包已经安装并加载：

```R
install.packages("factoextra")
library(factoextra)
```

下面是一个使用内置数据集 `iris` 进行 PCA 分析，并用 `fviz_screeplot()` 函数来绘制碎石图的例子：

```R
library(factoextra)
library(FactoMineR)

# 对 iris 数据集进行 PCA 分析
data(iris)
res.pca <- PCA(iris[,1:4], graph = FALSE)

# 绘制碎石图
fviz_screeplot(res.pca,
               addlabels = TRUE, # 在每个点添加标签
               ncp = 4,  # 显示前4个主成分的方差解释率
               main = "Scree Plot of PCA") # 添加主标题
```

在这个例子中，`iris[, 1:4]` 指定了用于 PCA 分析的数据集的列（特征）。`ncp = 4` 表示我们想要展示前4个主成分的结果。`addlabels = TRUE` 表示在每个碎石图上的点上添加标签，这样可以更清楚地看到每个主成分的方差解释率。`main` 参数用于设置图表的主标题。

## 聚类分析结果可视化

1. `fviz_dend()`：绘制树状图，显示**层次聚类结果**。
2. `fviz_cluster()`：绘制聚类结果的散点图。
3. `fviz_nbclust()`：绘制不同聚类数目下的评估指标曲线。

## 热图可视化

1. `fviz_heatmap()`：绘制矩阵数据的热图。

## 相关性分析结果可视化

1. `fviz_cor()`：绘制变量之间的相关性矩阵的热图。
2. `fviz_mca_ind()`：绘制个体在多元对应分析(MCA)空间中的投影图。
3. `fviz_mca_var()`：绘制变量在多元对应分析(MCA)空间中的投影图。

## 其他功能

1. `fviz_silhouette()`：绘制轮廓系数图，评估聚类结果的质量。
2. `fviz_cos2()`：绘制个体或变量对因子的贡献度。

以上是`factoextra`包中一些主要函数的介绍，按照功能进行分类。这些函数能够帮助您对因子分析和聚类分析的结果进行可视化和解释，从而更好地理解数据的结构和关系。您可以根据具体的分析任务使用相应的函数。



