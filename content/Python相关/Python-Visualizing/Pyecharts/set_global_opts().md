`set_global_opts()` 方法接受一个字典作为参数，该字典包含了各种全局图表选项。这些选项可以应用于图表中的所有组件，如标题、坐标轴、图例等。

### 常用属性
`set_global_opts()` 方法可以设置以下常用属性：
1. `title_opts`：设置标题选项，如标题文本、位置、样式等。
2. `legend_opts`：设置图例选项，如图例位置、布局方向、样式等。
3. `toolbox_opts`：设置工具箱选项，提供一些交互式操作，如数据视图、数据过滤等。
4. `tooltip_opts`：设置提示框选项，如触发类型、格式化器、位置等。
5. `xaxis_opts` 和 `yaxis_opts`：设置 X 轴和 Y 轴的坐标轴选项，如名称、标签、轴线样式等。

```python
from pyecharts.charts import Bar
from pyecharts import options as opts
# 创建一个 Bar 对象
bar = Bar()
# 使用 set_global_opts 方法设置全局选项
bar.set_global_opts(
    title_opts=opts.TitleOpts(title="水果销量"),
    xaxis_opts=opts.AxisOpts(name="水果"),
    yaxis_opts=opts.AxisOpts(name="销量"),
)
# 添加数据和配置
bar.add_xaxis(["苹果", "梨", "橘子", "香蕉"])
bar.add_yaxis("商店A", [5, 20, 36, 10])
bar.add_yaxis("商店B", [15, 6, 45, 20])
# 渲染图表到文件
bar.render('bar_chart.html')
```

### 使用举例
在上面的例子中，我们创建了一个 `Bar` 对象，并使用 `set_global_opts()` 方法设置了标题和坐标轴的全局选项。然后，我们添加了数据和配置，并渲染了图表到 HTML 文件。
通过使用 `set_global_opts()` 方法，您可以统一设置图表的全局样式和行为，使得图表更加美观和易用。
