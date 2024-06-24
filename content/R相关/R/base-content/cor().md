内置函数，得到相关性矩阵！

```R
cor(x, y = NULL, use = "everything", method = c("pearson", "kendall", "spearman"))
```

**参数介绍:**
- `x`: 一个数值型向量、矩阵或数据框。
如果是个**矩阵数据**，得到的就是变量两两之间的协方差对角矩阵！
```R
			Sepal.Length Sepal.Width Petal.Length Petal.Width
Sepal.Length 1.0000000 -0.1175698 0.8717538 0.8179411
Sepal.Width -0.1175698 1.0000000 -0.4284401 -0.3661259
Petal.Length 0.8717538 -0.4284401 1.0000000 0.9628654
Petal.Width 0.8179411-0.3661259 0.9628654 1.0000000
```

- `y`: 一个可选的数值型向量或矩阵。如果只提供了`x`,则计算`x`的相关矩阵。默认是NULL
- `use`: 指定如何处理缺失值。可以是"everything"(默认)、"all.obs"(删除所有含有缺失值的观测)、"complete.obs"(只保留完整观测)或"pairwise.complete.obs"(根据每对变量计算相关性)。
- `method`: 指定计算相关性系数的方法,可选"pearson"(皮尔逊相关)、"kendall"(肯德尔秩相关)或"spearman"(斯皮尔曼秩相关)。

**应用举例:**

1. 计算两个向量之间的皮尔逊相关系数:
```r
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)
cor(x, y)
```

2. 计算数据框各列之间的相关矩阵(皮尔逊相关):
```r
df <- data.frame(a = 1:5, b = 6:10, c = 11:15)
cor(df)
```

3. 计算数据框各列之间的斯皮尔曼秩相关矩阵:
```r
cor(df, method="spearman")
```

4. 当存在缺失值时,使用"pairwise.complete.obs"方式计算相关矩阵:
```r
df2 <- data.frame(a = c(1, 2, NA, 4, 5), b = c(6, NA, 8, 9, 10))
cor(df2, use = "pairwise.complete.obs")
```

总之,`cor()`函数是R语言中常用的计算相关性的函数,可以灵活地选择不同的计算方法和缺失值处理方式,满足各种数据分析需求。