`geom_tile()`函数是`ggplot2`包中用于创建**瓦片图（tile plot）或热图（heatmap）** 的函数。这种图形可以用来展示**两个分类变量之间的交叉和一个连续变量的大小**，通过不同的颜色来表示连续变量的值。

钻石的切割品质和颜色对价格的影响！
![[Pasted image 20240314110409.png]]


```r
geom_tile(mapping = NULL, data = NULL, stat = "identity", position = "identity",
          ..., na.rm = FALSE, show.legend = NA, inherit.aes = TRUE)
```

- **`mapping`**: 设置数据的美学属性，通常通过`aes()`函数来设置。对于`geom_tile()`来说，至少需要x和y轴的映射，通常还会映射填充颜色（`fill`）来表示连续变量的值。

指定fill填充色！

- **`data`**: 指定数据集。如果在`ggplot()`函数中已经指定，这里可以不设置。
- **`stat`**: 使用的统计变换，默认是`"identity"`，表示直接使用数据的值。
- **`position`**: 控制重叠的对象的调整，默认是`"identity"`，表示直接根据数据位置进行绘制。
- **`na.rm`**: 布尔值，是否移除`NA`值，默认为`FALSE`。
- **`show.legend`**: 逻辑值或`NA`，指定是否在图例中显示此图层，默认`NA`根据图层类型和情况自动判断。
- **`inherit.aes`**: 逻辑值，指定是否继承`ggplot()`中定义的美学映射，默认为`TRUE`。

### 应用举例

下面是使用`geom_tile()`函数创建热图的示例。假设我们有一个数据框`df`，它有三个列：两个分类变量`x`和`y`，以及一个与它们对应的连续变量`z`：

```r
library(ggplot2)

# 创建一个示例数据框
df <- expand.grid(x = 1:10, y = 1:10)
df$z <- runif(100)

# 使用ggplot和geom_tile创建热图
ggplot(df, aes(x = x, y = y, fill = z)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") + # 定义颜色渐变
  labs(title = "热图示例", x = "X轴", y = "Y轴", fill = "Z值") +
  theme_minimal() # 使用简洁的主题
```

在这个例子中，我们首先生成了一个包含x, y两个分类变量和一个连续变量z的数据框。`x`和`y`的值从1到10，`z`的值是从0到1的随机数。然后，我们使用`geom_tile()`绘制了热图，并通过`scale_fill_gradient()`设置了颜色的渐变，从白色（低值）到蓝色（高值）。这种图形非常适合于展示两个分类变量之间的关系以及它们对应的连续变量值的分布情况。


### 实际应用
用[[count()]]对两个分类变量进行计数。再绘制瓦片图。计数列n作为fill参数值！

```R
diamonds %>%
	count(color,cut)


输出
### A tibble:35 x 3
## color cut n
## <ord><ord> <int>
## 1 D Fair 163
##2 D Good 662
##3 D Very Good 1513
##4 D Premium 1603
## 5 D Ideal 2834
## Fair 6 E 224
##7 E Good 933
##8 E Very Good 2400
##9 E Premium 2337
## ]10 E Ideal 3903
## #. with 25 more rows


diamonds %>%
	count(color,cut) +
	ggplot(mapping=aes(x=color,y= cut)) +
		geom_tile(mapping= aes(fill = n))
```

