## umap()

![[Pasted image 20240421152340.png]]

Feature是初始维度变量，UMAP1和UMAP2是降维后的两个维度
Value是对应的某个高维度变量的值，用**颜色表示高维度数值**
这样整个图展示的就是高维度变量数值在低纬度（2维）上的分布结果，点越聚集颜色越近，该降维对高维度降维的效果越好！

```R
library(umap)
library(mclust)

data(banknote)
noteUmap <- banknote %>% 
  dplyr::select(-Status) %>% 
  as.matrix() %>% 
  umap(n_neibours = 7, min_dist = 0.1, 
       metric = "manhattan", n_epochs = 200, verbose = T)

# scale = F,是对函数scale中的scale参数进行调整；对data剔除c(-UMAP1, -UMAP2, -Status)相关列后进行操作
noteUmap1 <- banknote %>% 
  mutate_if(.funs = scale, .predicate = is.numeric, scale = F) %>% 
  mutate(UMAP1 = noteUmap$layout[,1], UMAP2 = noteUmap$layout[,2]) %>% 
  pivot_longer(c(-UMAP1, -UMAP2, -Status), names_to = "Feature", values_to = "Value", )


ggplot(noteUmap1, aes(UMAP1, UMAP2, col = Value, shape = Status)) +
  facet_wrap(~ Feature) +
  geom_point(size = 2) +
  scale_color_gradient(low = "darkblue", high = "cyan")

newNotes <- tibble::tibble(
  Length = c(214, 216),
  Left = c(130, 128),
  Right = c(132, 129),
  Bottom = c(12, 7),
  Top = c(12, 8),
  Diagonal = c(138, 142)
)
predict(noteUmap, newNotes)
```

`umap()` 函数在 R 语言中用于执行统一流形近似和投影（Uniform Manifold Approximation and Projection，简称UMAP）算法。UMAP 是一种流行的非线性降维技术，类似于 t-SNE，但通常更快并能**更好地保持全局结构**。它可以用于**可视化高维数据集**。

```r
install.packages("umap")
library(umap)
umap(data, n_neighbors=15, n_components=2, metric="euclidean", ...)
```

- `data`：输入数据。可以是矩阵或数据框，其中行表示样本，列表示特征。

- `n_neighbors`：考虑的**邻近点的数量**，默认值为15。这个参数控制UMAP的局部与全局结构之间的平衡。控制模糊搜索区域的半径，更少邻域到更多邻域；

越大越凸显全局结构？

- `n_components`：**目标维度数**，默认值为2。通常设置为2或3以便于可视化。

- n_epochs 优化步骤的**迭代次数**

- min_dist：低维下允许的**行间最小距离**，更集中到更分散；
越小，越凸显局部结构？

- `metric`：用于计算**距离的度量**，默认为"euclidean"（欧几里得距离）。UMAP支持多种距离度量，如"manhattan"、"cosine"等。


```r
# 加载umap包
library(umap)

# 假设data是你的高维数据
# 执行UMAP降维
umap_result <- umap(data, n_neighbors=30, n_components=2, metric="euclidean")

# 结果是一个包含降维坐标的数据框
# 使用这些坐标进行可视化
plot(umap_result$layout[,1], umap_result$layout[,2], xlab="UMAP 1", ylab="UMAP 2", main="UMAP Projection", pch=20, col=rainbow(nrow(data)))
```

这个简单的例子展示了如何对数据进行UMAP降维并将结果可视化。`umap_result$layout` 包含了降维后的坐标，可以用来绘制散点图或进行其他类型的可视化分析。