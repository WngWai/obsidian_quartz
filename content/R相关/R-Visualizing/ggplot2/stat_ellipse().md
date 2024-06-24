```R
ggplot(notePCA,aes(PCA1,PCA2,col = Status))+
  geom_point()+
  stat_ellipse(level = 0.90)+
  geom_point(data = newPCA, aes(PC1,PC2),col ='black',size = 4)
```

![[Pasted image 20240331164702.png]]


在R语言中，`stat_ellipse()`函数是`ggplot2`包中的一个统计函数，用于在散点图中**添加椭圆或椭球**。下面是`stat_ellipse()`函数的定义、参数介绍和一个应用举例：

**函数定义**：
```R
stat_ellipse(mapping = NULL, data = NULL, geom = "polygon", 
             type = "t", level = 0.95, segments = 51, 
             na.rm = FALSE, show.legend = NA, inherit.aes = TRUE, ...)
```

**参数介绍**：
- `mapping`：变量之间的映射关系。
- `data`：数据框或数据集。
- `geom`：椭圆或椭球的绘制类型，可选值为"polygon"（多边形）或"path"（路径）。
- `type`：椭圆或椭球的类型，可选值为"t"（t分布）或"norm"（正态分布）。
- `level`：椭圆或椭球的**置信水平**。
可以理解为椭圆囊括的散点范围

- `segments`：用于绘制椭圆或椭球的线段数量。
- `na.rm`：是否移除含有缺失值的观测。
- `show.legend`：是否显示图例。
- `inherit.aes`：是否继承图形参数。
- `...`：其他传递给`geom_polygon()`或`geom_path()`的参数。

**应用举例**：
以下是一个简单的应用示例，展示如何使用`stat_ellipse()`函数在散点图中添加椭圆：

```R
library(ggplot2)

# 创建散点图
p <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point()

# 添加椭圆
p + stat_ellipse()
```

在上述示例中，我们首先加载了`ggplot2`包，并使用`ggplot()`函数创建了一个散点图，其中x轴表示花萼长度（Sepal.Length），y轴表示花萼宽度（Sepal.Width），颜色根据鸢尾花的物种（Species）进行映射。

然后，我们使用`stat_ellipse()`函数在散点图中添加了椭圆，默认情况下，它将根据每个物种的均值和协方差矩阵绘制置信水平为0.95的椭圆。

运行代码后，散点图中将显示不同物种的数据点，并在每个物种的数据点周围添加椭圆。

这是`stat_ellipse()`函数的简单应用示例。您可以根据需要调整参数来自定义椭圆的样式和属性。