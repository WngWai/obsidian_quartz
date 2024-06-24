[ggplot2作图教程](https://zhuanlan.zhihu.com/p/370223674)

[ggplot2官方教程](https://ggplot2.tidyverse.org/articles/ggplot2.html)

- 数据dataset：作图用的原始数据；

- 美学 aes(): 几何或者统计对象的美学，比如位置，颜色，大小，形状等；

- 几何图形 geom_* :表示数据的**几何图形**；

- 主题 theme(): 图形的整体视觉默认值，如背景、网格、轴、默认字体、大小和颜色

- 刻度 scale_(): 数据与美学维度之间的映射，比如图形宽度的数据范围；

- 坐标系统 coord_: 数据的转换；

- 面 facet_: 数据图表的排列

- 统计转换 stat_: 数据的统计，对数据进行处理 ，比如百分位，拟合曲线或者和；
![Pasted image 20231005110243](Pasted%20image%2020231005110243.png)

![[24ea23f89560a22b341e94824b910ae.png]]

![[Pasted image 20240303143644.png]]


---
## 基础
### 基础图层：数据集+美学
[ggplot()](ggplot().md) 创建一个基本的绘图对象base(就是**画布**)！在画布的基础上再叠加各种要素。指定**数据**集，设置数据和图形的整体属性。在上面可以叠加多个图形

[aes()](aes().md) x,y轴映射关系，定义图形的映射关系。指定数据的变量与图形的视觉属性之间的对应关系，如 x 轴位置、y 轴位置、颜色、形状等。

### 几何图层：
真正意义上的在画布上呈现出来几个图形，geom_()

[color详情](color详情.md) color一般指边框色，fill指填充色

- 关系图（）

	主要为**多变量**间的关系

	- 连续型变量间
		
		[geom_point()](geom_point().md) 画散点图，**数据映射**
		[[geom_bin_2d()]]画矩形**分箱**图
		[[geom_hex()]]画六边形**分箱**图
		
		[geom_line()](geom_line().md) 画折线图
		
		[geom_smooth()](geom_smooth().md)画创建平滑拟合曲线，规避一些**离群极端点**

		另一个处理思路，是将一个连续型变量离散化！详见geom_boxplot()，如年龄和收入，将年龄离散化，实现更大维度的数据展示
		

	- 离散型变量间
		
		用**第三个变量（如圆圈尺寸、填充色等）** 表示两个离散型变量间的关系。
		
		[[geom_count()]]**计数图**

		[[geom_tile()]] **瓦片图（tile plot）或热图（heatmap）** 。seriation包用于**排序**，d3heatmap或者heatmaply包用于**更复杂**的瓦片图


-  分类图

	主要为**单个离散型变量**（分类变量）
	
	[geom_bar()](geom_bar().md) 画柱状图(条形图)，处理**分类型数据**
	
	[geom_col()](geom_col().md) 默认下直接用数值画柱状图


- 分布图

	主要为**单个连续性变量**。也可将两连续变量**中的一个变为分类变量**

	[geom_histogram()](geom_histogram().md) 画直方图，处理**数值型数据**
	
	[[geom_freqpoly()]] 分类+连续，画**频率多边形**，在一张图上**同时展示多个“条形图”**
	
	[geom_boxplot()](geom_boxplot().md) 连续，以及分类（用到cut_width()和cut_number()，cut()）+连续，箱线图

- 注释：

	[geom_text()](geom_text().md) 在**图形中**添加**文本标签**
	
	[[geom_label()]] 在**图形中**添加**文本标签**，多加了个**背景框**
	
	[[01-01 ggrepel为ggplot2的补充包]] 智能地**调整文本标签**的位置，是上面geom_text()和geom_label()的补充

- 辅助线
	[geom_vline()](geom_vline().md) 垂直线函数
	geom_hline()水平线
	geom_rect()画个矩形边框
	geom_segment()画个箭头


- 其他
	
	[[geom_jitter()]]添加**随机噪声到数据点**，以避免重叠点的可视化问题？
	
	[geom_density()](geom_density().md) 根据数值，计算**概率密度曲线**，概率密度分布
	
	[geom_function()](geom_function().md) 绘制自定义或匿名函数曲线，创建一个新的图形
	
	geom_emoji()**红包图形**
	
	[[geom_violin()]]  高质量的**统计图形**

### 统计图层
[stat_function()](stat_function().md)绘制自定义或匿名函数的曲线，通常与其他图层函数一起使用

[stat_summary()](stat_summary().md) 计算和绘制**统计摘要图形**。用于进行汇总统计，计算每组数据的统计摘要，如均值、中位数、标准差等，并将结果绘制为点、线、矩形等形状。

`stat_bin()`：用于进行直方图或柱状图的统计变换，将数据分组为离散的区间并计算每个区间的频数或密度。

`stat_smooth()`：用于拟合平滑曲线或回归曲线，根据数据的趋势绘制平滑的曲线，常用于数据的趋势分析和模型拟合。

`stat_boxplot()`：用于绘制箱线图，显示数据的分布情况，包括中位数、四分位数、异常值等。

[stat_density()](stat_density().md)：用于生成**密度图**，估计数据的概率密度函数，可用于分布形状的可视化。

`stat_ecdf()`：用于生成经验累积分布函数（empirical cumulative distribution function），展示数据的累积分布情况。

`stat_ellipse()`：用于绘**制椭圆**，显示二维数据的椭圆包络，常用于数据的相关性分析和聚类可视化。

### 度量/标度调整
虽然在绘图时这部分没有加，但都**默认值**放上去了！

[scale_x_continuous()](scale_x_continuous().md) 调整图形的x轴**刻度和修改刻度标签**

[scale_y_continuous()](scale_y_continuous().md)调整图形的y轴刻度和刻度标签

scale_x_1og10()后面用到再说

- `scale_color_*()`: 设置图形中**边框color**颜色的属性。

	- 离散型变量
	
		[scale_color_manual()](scale_color_manual().md)**手动调整**颜色映射
	
		scale_color_brewer() 将 Color Brewer 的色板应用到图形的**离散变量**上
	
		scale_color_hue() 为**离散变量**提供颜色映射，通过调整色调（hue）来生成颜色


	- 连续型变量

		scale_color_gradient() 为连续变量创建一个**两种颜色之间的简单渐变**
	
		[[scale_color_gradient2()]] 为连续变量创建一个**三色渐变**

		scale_color_viridis() 使用 viridis 色板为连续变量提供颜色映射。需要额外的[[01-02 viridis包（ggplot2扩展包）]]

- `scale_fill_*()`  设置图形中的**填充fill**颜色的属性

	- 离散型变量

		scale_fill_manual(): **手动**设置**离散变量的填充颜色**。可以指定每个离散值对应的具体颜色。
	
		[[scale_fill_hue()]]使用**色相环**来设置**离散**变量的填充颜色，颜色按照色相环的顺序分配。
	
		scale_fill_brewer(): 使用**RColorBrewer**包中的调色板来设置**离散**变量的填充颜色。
	

	- 连续型变量
	
		[[scale_fill_gradient()]] 使用**连续变量的值来创建填充颜色的渐变**。可以设置渐变的起始和结束颜色。
	
		scale_fill_gradientn(): 使用**连续**变量的值来创建填充颜色的渐变，可以指定多个颜色。
	
		scale_fill_gradient2(): 类似于scale_fill_gradient()，但允许指定**中间值**的颜色。

		[[scale_fill_distiller()]]适用于**连续变量**

		[[scale_fill_viridis()]] 基于virdis色板

	

- `scale_size_`

- `scale_alpha_`

- `scale_linetype_`

- `scale_alpha_`

### Coordinates坐标
`coord_*` ： 控制绘图的坐标系统和坐标轴的显示方式。它们允许您修改绘图的坐标轴刻度、范围和比例，以及坐标轴的显示方式

[coord_flip()](coord_flip().md) x和y轴**互换**

coord_fixed() X和Y轴**比例相同**

[coord_polar()](coord_polar().md)将**坐标系**转换为**极坐标系**

[coord_cartesian()](coord_cartesian().md) 约束**图形坐标显示的范围**。如果将ylim()函数单独列出来与geom_函数并列，是对原数据集**分组后的每组数量**进行了过滤

coord_map()

### 美化：辅助显示层
1，主题、标签：

[theme()](theme().md)设置图形的主题，包括背景、标题、轴标签、图例等的**外观和样式**，美化作用

[labs()](labs().md)设置图形的**标签**，包括**标题、轴标签、图例标题**等。

2，图例调整

[[guides()]]可以对单个图例进行详细操作，修改或指定图例（legend）和颜色条（color bar）的外观和位置。而theme()一般设置图例的全局属性

### 分面绘图
[facet_grid()](facet_grid().md) 基于x、y因子进行分面，如单x，单y，或x和y

[facet_wrap()](facet_wrap().md) 基于x、y、甚至**额外z**因子分面，如z1、z2...，分割x~y

### 图片保存
[[图形参数调整]]

[[ggsave()]]图表保存

---

## 进阶

[[gghalves包]]，时ggplot2的扩展包，将两个图拼在一起

[[patchwork包]] 拼图，将多个ggplot2的图片**组合**在一起，类似grid！


- 对复杂数据进行大致绘图

	能不能加个颜色？或者改变下背景格式？
	
	[[ggpairs()]]绘制多个变量之间关系的**散点图矩阵**，用于创建成对的散点图、箱线图和其他类型的图，以便于比较多个变量之间的关系。提供了更多的默认图形选项和更高的自定义能力，提供更全面的变量关系可视化
	
	[[ggscatmat()]]  也是**散点图矩阵**，重点在于**散点图**的可视化



gifski包

gapminder包，gapminder::gapminder数据

[[gganimate包]] 制作动态图形

配色包？

