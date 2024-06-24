ggscatmat(data, columns, color)

![[Pasted image 20240331161739.png]]

```R
library(GGally)
note<- as_tibble(mclust::banknote)
ggscatmat(note,columns =2:7,color="Status")
```
![[Pasted image 20240331162326.png]]


此函数为定量变量生成**散点图矩阵**，对角线上为**密度图**，上三角区域显示**相关系数**。

`ggscatmat()` 函数是 `GGally` 包中的一个函数，它也基于 `ggplot2`。重点在于**散点图的可视化**
```R
ggscatmat( data, columns = 1:ncol(data), color = NULL,  alpha = 1,  corMethod = "pearson")
``` 


参数  
data 一个数据矩阵。应包含数值（连续）数据。

columns 一个选项，用于选择原始数据集中**要使用的列**。默认为**1:ncol(data)**。

color 一个选项，用于根据**因子变量对数据进行分组**，并以不同的颜色表示。默认为NULL，即不进行颜色编码。如果提供了该选项，它将被转换为因子。

alpha 一个选项，用于设置大型数据的散点图**透明度**。默认为1。

corMethod 传递给cor的方法参数。

注意：原参数`corMethod`中的默认值“pearson”可能是一个拼写错误，通常应该是“pearson”或“pearson”，代表皮尔逊相关系数（Pearson correlation coefficient）。这里我假设它应该是“pearson”，但请根据实际情况进行确认。