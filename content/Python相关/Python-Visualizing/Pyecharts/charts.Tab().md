在 `pyecharts` 中，`Tab` 类用于创建一个标签页容器，可以包含多个标签页，每个标签页可以包含一个或多个图表。这通常用于创建一个报告或仪表板，其中包含多个相关的图表，每个图表都位于不同的标签页中。

在多个选项中进行切换。
![[Pasted image 20240221201734.png]]
### 定义
`Tab` 类是 `pyecharts` 中的一个类，用于创建一个标签页容器。
```python
from pyecharts.charts import Tab
# 创建一个 Tab 对象
tab = Tab()
```
### 常用属性
`Tab` 类具有以下常用属性：
1. `title`：设置标签页的标题。
2. `add`：向标签页添加一个子组件，通常是图表或其他页面元素。
### 使用举例
```python
from pyecharts.charts import Bar, Line, Tab
from pyecharts import options as opts
# 创建一个 Tab 对象
tab = Tab()
# 创建一个 Bar 对象
bar = Bar()
bar.add_xaxis(["苹果", "梨", "橘子", "香蕉"])
bar.add_yaxis("商店A", [5, 20, 36, 10])
bar.add_yaxis("商店B", [15, 6, 45, 20])
bar.set_global_opts(title_opts=opts.TitleOpts(title="水果销量"))
# 创建一个 Line 对象
line = Line()
line.add_xaxis(["1月", "2月", "3月", "4月"])
line.add_yaxis("销售额", [820, 932, 901, 934])
line.set_global_opts(title_opts=opts.TitleOpts(title="销售额"))
# 将 Bar 和 Line 对象添加到 Tab 对象中
tab.add(bar, "商店A销量")
tab.add(line, "销售额趋势")
# 渲染标签页到文件
tab.render('tab_chart.html')
```
在这个示例中，我们创建了一个 `Tab` 对象，并向其中添加了两个图表：一个 `Bar` 图表和一个 `Line` 图表。每个图表都位于不同的标签页中。然后，我们使用 `render` 方法将整个标签页渲染为一个 HTML 文件。通过这种方式，您可以创建一个包含多个标签页的页面，每个标签页中包含一个或多个图表，用于展示和分析数据。
