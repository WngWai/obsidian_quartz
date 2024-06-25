在Python的Pyecharts库中，`Gauge()`函数用于创建仪表盘（Gauge）图表，它可以显示一个值在一个范围内的指示器位置。

**函数定义**：
```python
Gauge.init(title, subtitle='', **kwargs)
```

**参数**：
以下是`Gauge()`函数中常用的参数：

- `title`：仪表盘图表的标题，类型为str。

- `subtitle`（可选）：仪表盘图表的副标题，类型为str，默认为空字符串。

- `**kwargs`：其他可选参数，用于自定义仪表盘的样式和行为。常用的可选参数包括：
  - `title_pos`：标题的位置，可选值为'auto'（默认值，自动选择位置）、'left'、'right'、'center'。
  - `title_top`：标题的垂直位置，可以是像素值或百分比。
  - `title_color`：标题的颜色。
  - `title_text_size`：标题的字体大小。
  - `subtitle_text_size`：副标题的字体大小。
  - `tooltip_formatter`：提示框内容格式化函数。
  - `legend_selectedmode`：图例的选择模式，可选值为'single'（单选）或'multiple'（多选）。
  - `radius`：仪表盘的半径大小，可以是像素值或百分比。
  - `start_angle`：仪表盘的起始角度，可选值为0到360之间的任意角度。
  - `end_angle`：仪表盘的结束角度，可选值为0到360之间的任意角度。
  - `split_number`：刻度的分割数量。
  - `axis_line_color`：仪表盘轴线的颜色。
  - `axis_line_width`：仪表盘轴线的宽度。
  - `axis_label_color`：刻度标签的颜色。
  - `pointer_color`：指针的颜色。
  - `detail_text_color`：指示器详细信息的颜色。
  - `detail_text_size`：指示器详细信息的字体大小。
  
**示例**：
以下是使用`Gauge()`函数创建仪表盘图表的示例：

```python
from pyecharts import options as opts
from pyecharts.charts import Gauge

# 创建一个仪表盘图表实例
gauge = (
    Gauge()
    .add("业务指标", [("完成率", 66.6)])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="仪表盘示例"),
        legend_opts=opts.LegendOpts(is_show=False),
    )
)

# 渲染图表
gauge.render("gauge.html")
```

在上述示例中，我们首先导入了`pyecharts.options`和`pyecharts.charts`模块，然后创建了一个`Gauge()`函数的实例。

通过`.add()`方法，我们向仪表盘图表添加了一个指示器，指示器的名称是"完成率"，值为66.6。

然后，使用`.set_global_opts()`方法设置了图表的全局选项，包括标题和图例。

最后，我们使用`.render()`方法将图表渲染为HTML文件。

通过运行上述代码，我们可以创建一个简单的仪表盘图表，并将其保存为HTML文件。实际上，我们可以使用其他方法和选项来自定义仪表盘的样式和行为，以满足特定的需求。

请注意，上述示例仅演示了基本用法，更多详细的参数和选项可以参考Pyecharts官方文档。