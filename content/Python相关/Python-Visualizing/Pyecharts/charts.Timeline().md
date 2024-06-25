在Python的Pyecharts库中，`Timeline()`函数用于创建时间轴（Timeline）图。
![[Pasted image 20240221083539.png]]
**函数定义**：
```python
pyecharts.charts.Timeline()
```
**参数**：
以下是`Timeline()`函数中常用的参数：
- `init_opts`（可选）：初始化选项，用于配置图表的基本属性，如标题、背景颜色等。默认值为`opts.InitOpts()`。
- `width`（可选）：图表的宽度。可以是像素值（如`"800px"`）或百分比（如`"80%"`）。默认值为`"800px"`。
- `height`（可选）：图表的高度。可以是像素值（如`"600px"`）或百分比（如`"60%"`）。默认值为`"600px"`。
**示例**：
以下是使用`Timeline()`函数创建时间轴图的示例：
```python
from pyecharts.charts import Timeline, Bar
from pyecharts import options as opts

# 创建时间轴对象
timeline_chart = Timeline()

# 添加数据
data1 = [("A", 10), ("B", 20), ("C", 30)]
data2 = [("A", 20), ("B", 30), ("C", 40)]
data3 = [("A", 30), ("B", 40), ("C", 50)]

# 创建柱状图对象
bar_chart = Bar()

# 设置柱状图的标题和数据
bar_chart.add_xaxis(["A", "B", "C"])
bar_chart.add_yaxis("Series 1", data1)
bar_chart.add_yaxis("Series 2", data2)
bar_chart.add_yaxis("Series 3", data3)

# 添加柱状图到时间轴中
timeline_chart.add(bar_chart, "Time 1")
timeline_chart.add(bar_chart, "Time 2")
timeline_chart.add(bar_chart, "Time 3")

# 渲染图表到HTML文件中
timeline_chart.render("timeline_chart.html")
```

在上述示例中，我们首先导入了`Timeline`类和相关的配置项模块`opts`。

然后，我们创建了一个`Timeline`对象，并设置了图表的标题等。

接下来，我们创建了一个柱状图对象`bar_chart`，并设置了柱状图的数据。

然后，我们使用`add()`方法将柱状图添加到时间轴中，每个时间点对应一个柱状图。

最后，我们使用`render`函数将图表渲染到一个HTML文件中。可以打开生成的`timeline_chart.html`文件来查看时间轴图的效果。

除了上述示例中的参数，`Timeline()`函数还有其他可用的参数和配置选项，用于自定义时间轴图的样式、标签、颜色等。您可以根据您的需求进行进一步的定制和配置。