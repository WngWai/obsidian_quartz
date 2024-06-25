在Pyecharts库中，`Boxplot()`函数用于创建箱线图（Boxplot chart）。

**函数定义**：
```python
Boxplot(init_opts: Union[opts.InitOpts, dict] = <pyecharts.options.global_options.InitOpts object>, **kwargs)
```

**参数**：
以下是`Boxplot()`函数中常用的参数：

- `init_opts`：可选参数，初始化配置项。可以是`opts.InitOpts`类型的对象或包含初始化配置的字典。

- `**kwargs`：其他可选参数，用于设置箱线图的样式、数据等。

**返回值**：
`Boxplot()`函数返回一个`Boxplot`类的实例，用于绘制箱线图。

**示例**：
以下是使用`Boxplot()`函数绘制箱线图的示例：

```python
from pyecharts import options as opts
from pyecharts.charts import Boxplot

# 创建箱线图
boxplot = Boxplot()

# 设置箱线图的标题和数据
boxplot.set_global_opts(title_opts=opts.TitleOpts(title="Boxplot Chart Example"))
boxplot.add_xaxis(["A", "B", "C"])
boxplot.add_yaxis("Category 1", [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
boxplot.add_yaxis("Category 2", [[2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]])

# 渲染生成HTML文件
boxplot.render("boxplot_chart.html")
```

在上述示例中，我们首先导入了所需的模块和类。

然后，我们使用`Boxplot()`函数创建了一个箱线图对象`boxplot`。

接着，我们使用`set_global_opts()`方法设置了箱线图的标题为"Boxplot Chart Example"。

使用`add_xaxis()`方法设置了x轴的刻度标签，这里设置为["A", "B", "C"]。

使用`add_yaxis()`方法添加了两个分类的数据到箱线图。每个分类的数据是一个二维列表，其中每一行表示一个箱线图的盒须区间。在示例中，我们创建了两个分类的数据，每个分类包含了三个箱线图的盒须区间。

最后，我们使用`render()`方法将箱线图渲染为HTML文件。

请注意，上述示例仅演示了基本用法，更多详细的参数和选项可以参考Pyecharts的官方文档。