在Python的Pyecharts库中，`Scatter()`函数用于创建散点图（Scatter Chart）。

**函数定义**：
```python
pyecharts.charts.Scatter()
```

**参数**：
以下是`Scatter()`函数中常用的参数：

- `init_opts`（可选）：初始化选项，用于配置图表的基本属性，如标题、背景颜色等。默认值为`opts.InitOpts()`。

- `page_title`（可选）：图表所在HTML页面的标题。默认值为`"Scatter Chart"`。

- `width`（可选）：图表的宽度。可以是像素值（如`"800px"`）或百分比（如`"80%"`）。默认值为`"800px"`。

- `height`（可选）：图表的高度。可以是像素值（如`"600px"`）或百分比（如`"60%"`）。默认值为`"600px"`。

* `Scatter()`：创建一个散点图对象，可以使用`add_yaxis()`方法添加一个或多个系列的数据，也可以使用`set_symbol_size()`方法设置散点的大小，或者使用`set_symbol()`方法设置散点的形状。
**示例**：
以下是使用`Scatter()`函数创建散点图的示例：

```python
from pyecharts.charts import Scatter
from pyecharts import options as opts

# 创建散点图对象
scatter_chart = Scatter()

# 设置图表标题和大小
scatter_chart.set_global_opts(
    title_opts=opts.TitleOpts(title="散点图示例"),
    visualmap_opts=opts.VisualMapOpts(),
    legend_opts=opts.LegendOpts(is_show=True),
)

# 添加数据
scatter_chart.add_xaxis(["A", "B", "C", "D", "E"])
scatter_chart.add_yaxis("系列1", [10, 20, 30, 40, 50])
scatter_chart.add_yaxis("系列2", [50, 40, 30, 20, 10])

# 渲染图表到HTML文件中
scatter_chart.render("scatter_chart.html")
```

在上述示例中，我们首先导入了`Scatter`类和相关的配置项模块`opts`。

然后，我们创建了一个`Scatter`对象，并设置了图表的标题、可视化映射选项和图例选项等。

接下来，我们添加了X轴的数据和两个Y轴系列的数据。

最后，我们使用`render`函数将图表渲染到一个HTML文件中。可以打开生成的`scatter_chart.html`文件来查看散点图的效果。

除了上述示例中的参数，`Scatter()`函数还有其他可用的参数和配置选项，用于自定义散点图的样式、标签、颜色等。您可以根据您的需求进行进一步的定制和配置。