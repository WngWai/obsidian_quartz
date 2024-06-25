在Python的Pyecharts库中，`Pie()`函数用于创建饼图（Pie Chart）。

**函数定义**：
```python
pyecharts.charts.Pie()
```

**参数**：
以下是`Pie()`函数中常用的参数：

- `init_opts`（可选）：初始化选项，用于配置图表的基本属性，如标题、背景颜色等。默认值为`opts.InitOpts()`。

- `page_title`（可选）：图表所在HTML页面的标题。默认值为`"Pie Chart"`。

- `width`（可选）：图表的宽度。可以是像素值（如`"800px"`）或百分比（如`"80%"`）。默认值为`"800px"`。

- `height`（可选）：图表的高度。可以是像素值（如`"600px"`）或百分比（如`"60%"`）。默认值为`"600px"`。

* `Pie()`：创建一个饼图对象，可以使用`add()`方法添加一个系列的数据，也可以使用`set_colors()`方法设置饼图的颜色，或者使用`set_series_opts()`方法设置标签，高亮等选项。

**示例**：
以下是使用`Pie()`函数创建饼图的示例：

```python
from pyecharts.charts import Pie
from pyecharts import options as opts

# 创建饼图对象
pie_chart = Pie()

# 设置图表标题和大小
pie_chart.set_global_opts(
    title_opts=opts.TitleOpts(title="饼图示例"),
    legend_opts=opts.LegendOpts(is_show=True),
)

# 添加数据
data = [("A", 10), ("B", 20), ("C", 30), ("D", 40), ("E", 50)]
pie_chart.add("", data)

# 渲染图表到HTML文件中
pie_chart.render("pie_chart.html")
```

在上述示例中，我们首先导入了`Pie`类和相关的配置项模块`opts`。

然后，我们创建了一个`Pie`对象，并设置了图表的标题和图例选项等。

接下来，我们添加了饼图的数据，数据以列表形式表示，每个元素是一个二元组，包含饼图的名称和对应的数值。

最后，我们使用`render`函数将图表渲染到一个HTML文件中。可以打开生成的`pie_chart.html`文件来查看饼图的效果。

除了上述示例中的参数，`Pie()`函数还有其他可用的参数和配置选项，用于自定义饼图的样式、标签、颜色等。您可以根据您的需求进行进一步的定制和配置。