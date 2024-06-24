`geom_hex()` 是 `ggplot2` 包中的一个函数，用于在二维空间内通过**六边形分箱**来可视化数据点的密度。这种方法特别适用于大数据集，因为它可以有效地揭示数据点的聚集区域，而不会因为点过多而导致的过度绘图（overplotting）问题。


![[Pasted image 20240314113125.png]]


```R
geom_hex(mapping = NULL, data = NULL, stat = "hex", position = "identity", ..., na.rm = FALSE, show.legend = NA, inherit.aes = TRUE)
```

- `mapping`：设置数据的美学映射，通常在 `aes()` 函数中指定。
- `data`：指定使用的数据集。如果在 `ggplot()` 函数中已指定，则此参数可以省略。
- `stat`：使用的统计变换，默认为 `"hex"`，表示使用六边形分箱统计。
- `position`：设置图层的位置调整。对于 `geom_hex()` 来说，通常保留默认值 `"identity"`。
- `na.rm`：一个逻辑值，指示是否移除包含缺失值的数据，默认为 `FALSE`。
- `show.legend`：逻辑值，指定是否在图例中显示该图层，默认根据情况自动判断。
- `inherit.aes`：逻辑值，指定是否继承全局映射，默认为 `TRUE`。
- `...`：其他参数，如 `binwidth` 或者 `bins`，这些参数可以直接传递给 `stat_hex()` 来控制六边形的大小或数量。

### 应用举例

假设我们有一组数据，想要可视化这些数据在二维空间中的分布情况。以下是使用 `geom_hex()` 函数的一个简单示例：

```R
# 加载ggplot2包
library(ggplot2)

# 生成示例数据
set.seed(123)
data <- data.frame(x = rnorm(10000), y = rnorm(10000))

# 使用 geom_hex 创建六边形密度图
ggplot(data, aes(x = x, y = y)) + 
  geom_hex() + 
  scale_fill_viridis_c() +  # 使用viridis颜色映射
  labs(title = "二维数据的六边形密度图", x = "X轴", y = "Y轴")
```

在这个例子中，我们首先生成了一个由两个正态分布随机变量 `x` 和 `y` 组成的数据框 `data`。然后，我们使用 `geom_hex()` 函数在二维空间中创建了一个六边形密度图，其中 `scale_fill_viridis_c()` 函数用于应用一个颜色渐变，以更清晰地显示密度的不同级别。最终，我们通过 `labs()` 函数添加了图表的标题和轴标签。

`geom_hex()` 是一种强大的函数，非常适合于探索和展示大型数据集中的二维空间分布，它通过聚合相近的点到六边形单元内，帮助我们识别数据的分布模式。