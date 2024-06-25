在 `pyecharts` 中，`Page` 类用于创建一个页面，该页面**可以包含多个图表**。这通常用于创建一个报告或仪表板，其中包含多个相关的图表。
![[Pasted image 20240221172129.png]]
`Page` 类是 `pyecharts` 中的一个类，用于创建一个页面容器，可以包含多个子组件，如图表和其他文本元素。
```python
from pyecharts.charts import Page
# 创建一个 Page 对象
page = Page()
```

1. `title`：设置页面的标题。
2. `theme`：设置图表的主题，如 'light'（浅色）、'dark'（深色）等。
3. `add`：向页面添加一个子组件，通常是图表或其他页面元素。

### 使用举例
```python
from pyecharts.charts import Bar, Line
from pyecharts import options as opts
# 创建一个 Page 对象
page = Page()
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

# 将 Bar 和 Line 对象添加到 Page 对象中
page.add(bar)
page.add(line)
# 渲染页面到文件
page.render('page_chart.html')
```
在这个示例中，我们创建了一个 `Page` 对象，并向其中添加了两个图表：一个 `Bar` 图表和一个 `Line` 图表。然后，我们使用 `render` 方法将整个页面渲染为一个 HTML 文件。通过这种方式，您可以创建一个包含多个图表的页面，用于展示和分析数据。

### 动态布局
怎么用纯pyecharts实现大屏可视化？ - Summer是只白喵的回答 - 知乎
https://www.zhihu.com/question/386723186/answer/2273228042

```python
page = Page(layout=Page.DraggablePageLayout, page_title="2020东京奥运会奖牌榜") 

# 在页面中添加图表
page.add(title(), 
		 map_world(),
		 bar_medals(),
		 pie_china(),) 

page.render('test.html')
```

对图片布局完毕后，要记得点击左上角“save config”对布局文件进行保存。
点击后，本地会生成一个chart_config.json的文件，这其中包含了每个图表ID对应的布局位置。
调用保存好的布局文件，重新生成html。

```python
page.save_resize_html('test.html', 
					  cfg_file='chart_config.json',
					  dest='奥运.html')
```
