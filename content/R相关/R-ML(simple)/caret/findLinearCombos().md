`findLinearCombos`用于**检测矩阵或数据框中的线性组合特征**。线性组合特征是指可以通过其他特征的线性组合表示的特征，这些特征可能导致共线性问题，并影响模型的稳定性和性能。

```R
findLinearCombos(x)
```

- **`x`**: 一个数x值矩阵或数据框，用于**检测线性组合特征**。这是该函数唯一的必需参数，输入的数据集应仅包含数值型数据。

`findLinearCombos` 函数返回一个列表，包含以下两个元素：

- **`linearCombos`**: 一个列表，每个元素是一个向量，表示**可以线性组合的特征索引**。
- **`remove`**: 一个向量，包含**建议移除的特征的索引**。这些特征通常是冗余的，因为它们可以由其他特征线性组合得到。


5（Cmb）可以由 1（Sepal.Length）和4（Petal.Width）线性组合得到！
只能排除严谨的线性关系！

```r
library(tidyverse)
library(caret)

iris_numeric <- iris[1:4]# 初始的基本集iris
new_iris <- iris_numeric

# 人为添加两列
new_iris$Cmb<- 6.7*new_iris$Sepal.Length - 0.9*new_iris$Petal.Width
set.seed(68)
new_iris$Cmb.N <- new_iris$Cmb + rnorm(nrow(new_iris), sd=0.1)
options(digits = 4)

# 输出查看
head(new_iris, n = 3)
# 输出内容
  Sepal.Length Sepal.Width Petal.Length Petal.Width   Cmb
1          5.1         3.5          1.4         0.2 33.99
2          4.9         3.0          1.4         0.2 32.65
3          4.7         3.2          1.3         0.2 31.31
  Cmb.N
1 34.13
2 32.63
3 31.27

# 查找查看
findLinearCombos(new_iris)
# 输出内容
$linearCombos
$linearCombos[[1]]
[1] 5 1 4

$remove
[1] 5

```





