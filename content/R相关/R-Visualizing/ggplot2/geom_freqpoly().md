`geom_freqpoly()`是`ggplot2`包中的一个函数，用于绘制**频率多边形图**。这种类型的图表通常用于展示数据的分布情况，类似于直方图，但它通过**连接各个直方图顶点的线条**来表示，从而提供了一种稍微不同的视角来查看数据的分布。


![[Pasted image 20240312142819.png]]


```r
geom_freqpoly(mapping = NULL, data = NULL, stat = "bin", position = "identity",
              ..., na.rm = FALSE, show.legend = NA, inherit.aes = TRUE)
```

### 参数含义

- `mapping`：设置图层的美学属性，如x、y轴。通常通过`aes()`函数来定义。
- `data`：指定绘图所使用的数据集。
- `stat`：用于计算图层统计变换，默认是`"bin"`，表示数据将会被分组并计算每组的计数。
- `position`：设置图层位置的调整。对于`geom_freqpoly`来说，默认值是`"identity"`，因为分组计数通常不需要位置调整。
- `...`：其他参数和图层特有的美学属性。
- `na.rm`：逻辑值，如果设置为`TRUE`，则会移除`NA`值。
- `show.legend`：逻辑值或`NA`，控制这一图层是否应该在图例中表示。
- `inherit.aes`：逻辑值，如果是`TRUE`（默认），则这一图层将继承`ggplot()`函数中定义的全局美学映射。

### 应用举例

以下是一个如何使用`geom_freqpoly()`函数的例子：

```r
# 加载ggplot2包
library(ggplot2)

# 创建一个简单的数据集
data <- data.frame(
  values = c(rnorm(100), rnorm(100, mean = 3, sd = 1.5))
)

# 使用ggplot2绘制频率多边形图
ggplot(data, aes(x = values)) +
  geom_freqpoly(binwidth = 0.5, color = "blue") +
  ggtitle("Frequency Polygon")
```

在这个例子中：

- 首先，创建一个包含两组正态分布数据的数据框`data`。
- 然后，使用`ggplot()`函数初始化一个图像对象，设置数据集为`data`，并通过`aes()`函数定义美学映射，指定`x`轴为`values`字段。
- 使用`geom_freqpoly()`函数添加频率多边形图层，设置`binwidth`为0.5来控制分组宽度，并将多边形线条颜色设置为蓝色。
- 最后，通过`ggtitle()`添加图表标题。

这个例子展示了如何使用`geom_freqpoly()`绘制数据的分布情况，通过调整`binwidth`参数，可以控制多边形的精细程度。