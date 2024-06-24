`findCorrelation`函数用于检测具有高相关性的特征，并建议移除其中一些特征以减少多重共线性。

```R
findCorrelation(cor_matrix, cutoff)
```

**主要参数:**

- **`cor_matrix`**: 一个**相关性矩阵**。
```R
# 需要剔除DF中的非数值型变量
cor_matrix  <- cor(cardata_df %>% select(-class))

# 可以看相关性矩阵的图形
library(corrplot)
corrplot(cor_matrix, method = "color", 
         tl.cex = 0.8, tl.col = "black")
```


- **`cutoff`**: 一个数值，指定相关性阈值（默认是**0.90**）。高于此阈值的特征对将被视为高度相关。
- **`verbose`**: 布尔值，是否打印详细信息（默认为FALSE）。

**返回值:**

- 返回一个向量，包含**建议移除的特征的索引**。默认是高于0.9的相关性变量，建议移除！


```R
library(tidyverse)
library(caret)

iris_numeric <- iris[1:4]# 初始的基本集iris

# 看协方差矩阵
cor(iris_numeric)
		Sepal.Length Sepal.Width Petal.Length Petal.Width
Sepal.Length 1.0000000 -0.1175698 0.8717538 0.8179411
Sepal.Width -0.1175698 1.0000000 -0.4284401 -0.3661259
Petal.Length 0.8717538 -0.4284401 1.0000000 0.9628654
Petal.Width 0.8179411-0.3661259 0.9628654 1.0000000

# 建议移除的特征的索引
iris_cor <- cor(iris_numeric)
findCorrelation(iris_cor)
[1]3

findcorrelation(iris_cor,cutoff=0.99)
integer(0)

findCorrelation(iris_cor,cutoff=0.80)
[1] 3 4

```