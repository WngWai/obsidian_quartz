**交互式图表和可视化组件。比如数据面板！**
[强推：Pyecharts功能网页](https://pyecharts.org/#/zh-cn/intro)

python中的Pyecharts包是一个**基于Echarts**的数据可视化库，它允许用户使用Python语言生成各种类型的交互式图表和数据可视化，**偏向于前端**。
Pyecharts强大的**数据交互功能**，使数据表达信息更加生动，增加了人机互动效果，并且数据呈现效果可直接导出为html文件，增加数据结果交互的机会，使得信息沟通更加容易。

### 图表类
```python
from pyecharts.charts import Bar
from pyecharts import options as opts

# 链式调用
# 你所看到的格式其实是 `black` 格式化以后的效果
# 可以执行 `pip install black` 下载使用
bar = (
    Bar()
    .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
    .set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
    # 或者直接使用字典参数
    # .set_global_opts(title_opts={"text": "主标题", "subtext": "副标题"})
)
bar.render()

# 不习惯链式调用的开发者依旧可以单独调用方法
bar = Bar()
bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
bar.set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
bar.render()
```

图表类的公共属性：
init：初始化图表的基本配置，如标题、图例、工具箱等。
options：一个包含所有图表选项的字典。
xaxis：X轴的配置项。
yaxis：Y轴的配置项。
series：图表的数据系列。
title：标题的配置项。
legend：图例的配置项。
toolbox：工具箱的配置项。
tooltip：提示框的配置项。

图表类的公共方法：
add_xaxis()向图表中**添加X轴的数据**。
add_yaxis()向图表中**添加Y轴的数据和系列名称**。

[[set_global_opts()]]设置**全局的图表选项**，如标题、图例、工具箱等。
[[set_series_opts()]]设置所有数据系列的**通用选项**。

add_jsfuncs：添加自定义的JavaScript函数，用于在图表中执行特定的操作。

render()将**图表渲染为HTML**，返回一个HTML文件。不写路径，默认保存为render.html
render_notebook在**Jupyter Notebook中直接渲染**图表。

图表类：

[[charts.Bar()]]创建**柱状**图，支持多类别、堆叠柱状图等。

[[charts.Line()]]创建**折线**图，支持多条线、面积图、平滑曲线等。

[[charts.Scatter()]]创建**散点**图，可以展示两个连续变量之间的关系。

[[charts.Pie()]]创建饼图，展示不同类别的占比情况。

HeatMap()创建热力图，用于展示变量之间的关联程度。

[[charts.Map()]]创建地理地图，支持世界地图、中国地图等，并可以对地图上的区域进行着色、标记。

[[charts.WordCloud()]]创建**词云图**，用于展示文本数据中关键词的频率和重要性。

[[charts.Gauge()]]创建仪表盘图，用于展示单个指标的变化情况。


[[charts.Graph()]]创建关系图，用于展示节点和边之间的关系。


[[charts.Tree()]]创建树图。用于展示层次结构或树形数据。

[[charts.Boxplot()]]创建箱线图。用于展示数据的分布情况。

[[charts.EffectScatter()]]创建动态散点图。用于展示地理数据和时间的关系。

`Parallel`：创建平行坐标图。用于展示多个维度之间的关系。
`Sankey`：创建桑基图。用于展示流量、能量等的流动情况。
`Radar`：创建雷达图。用于展示多个变量之间的对比情况。

### 组合类（布局类）
用于将不同类型的图表组合在一起，实现**多图展示或叠加效果**。
```python
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Page

# 创建一个Page对象，并启用DraggablePageLayout
page = Page(layout=Page.DraggablePageLayout, page_title="我的报告")

# 添加图表
bar = Bar()
bar.add_xaxis(["苹果", "梨", "橘子", "香蕉", "樱桃"])
bar.add_yaxis("水果店A", [5, 20, 36, 10, 75])
bar.set_global_opts(title_opts=opts.TitleOpts(title="水果销量"))

line = Line()
line.add_xaxis(["苹果", "梨", "橘子", "香蕉", "樱桃"])
line.add_yaxis("水果店A", [5, 20, 36, 10, 75])
line.set_global_opts(title_opts=opts.TitleOpts(title="水果销量"))

# 组合
page.add(bar)
page.add(line)
# page.add(bar, line)

# 渲染Page到HTML
page.render("my_draggable_page.html")
```

???
```python
# 将拖拽生成的json文件，跟原html相结合，生成新的html文件
page.save_resize_html('my_draggable_page.html', 
					  cfg_file='chart_config.json',
					  dest='new.html')
```

[[charts.Page()]]**顺序多图**，创建一个**页面**，可以使用`add()`方法将一个或多个图表添加到页面。通常用于创建一个报告或仪表板，其中包含多个相关的图表

[[charts.Grid()]]**并行多图**，创建一个**网格对象**，可以使用`add()`方法将一个或多个图表对象添加到网格中，也可以使用`set_global_opts()`方法设置网格的全局选项。

[[charts.Timeline()]]**时间线轮播多图**，创建一个**时间轴对象**，可以使用`add()`方法将一个或多个图表对象添加到时间轴中，实现图表的动态展示效果，如根据时间变化显示不同的数据等。可以展示数据随时间的变化趋势

charts.Tab()**选项卡**多图。也是用add()方法，实现图表的切换

[[charts.Overlap()]]创建一个**重叠对象**，可以使用`add()`方法将一个或多个图表对象添加到重叠中，实现图表的叠加效果，如在折线图上叠加散点图等。

### 全局配置类options
```python
from pyecharts import options as opts
```

设置**图表的全局配置项**。在创建图表对象或添加数据时，使用这些选项类来设置图表的各种属性，如标题，图例，坐标轴、图表的样式、布局、标签、工具箱等。

[[opts.AxisOpts()]]**坐标轴配置**，如轴类型、刻度标签、轴线样式等。
[[opts.TitleOpts()]]**标题配置**，如标题文本、位置、字体样式等。
[[opts.LegendOpts()]]**图例**配置，如图例位置、显示隐藏、字体样式等。


[[opts.TooltipOpts()]]设置**提示框**的选项，如触发方式、显示内容、样式等。

4. `ToolboxOpts`：设置**工具箱**的选项，如显示工具箱、工具箱图标、功能按钮等。
5. `VisualMapOpts`：设置**视觉映射**的选项，如颜色映射、范围、标签等。

7. `GridOpts`：用于设置图表的**网格**选项，如背景色、边距、布局等。
8. `DataZoomOpts`：用于设置**数据缩放**选项，如启用缩放、缩放范围、滑动条样式等。
9. `TextStyleOpts`：用于设置**文本样式**的选项，如字体、颜色、对齐方式等。
10. `LineStyleOpts`：用于设置**线条样式**的选项，如线宽、颜色、类型等。
11. `ItemStyleOpts`：用于设置**图元样式**的选项，如颜色、透明度、边框样式等。

### 其他类
主题类：用于设置图表的主题风格，如 ThemeType（内置主题类型），Theme（自定义主题）等。每个主题类都有自己的参数和方法，用于设置主题的颜色，字体，背景等。

`Theme()`：创建一个主题对象，可以使用`add()`方法添加一个或多个自定义的主题，也可以使用内置的主题，如`ThemeType.LIGHT`，`ThemeType.DARK`等，然后在创建图表对象时，使用`init_opts`参数指定主题，实现图表的风格切换。

### 工具类：

用于提供一些辅助功能，如 Faker（伪造数据集），JsCode（JavaScript 代码），SymbolType（标记类型）等。每个工具类都有自己的参数和方法，用于生成或转换一些数据或代码。


### 与matplotlib的区别
可能更加强调前端，动态图。

1. 语法风格：Pyecharts是基于Echarts绘图库的Python封装，它提供了一种**更直观、易于使用的API**，使用者可以通过**链式调用**方法来构建图表。而matplotlib则是一个较为**传统的绘图库**，使用者需要**逐步调用函数**来创建和设置图形。
2. 功能和图表类型：Pyecharts提供了丰富的**图表类型**，包括折线图、柱状图、饼图、散点图等，还支持地理**数据可视化、动态图、3D图**等高级功能。而matplotlib也提供了各种常见的图表类型，但**在高级功能方面相对较弱**。
3. 可视化效果：Pyecharts基于Echarts，拥有强大的可定制性和交互性，可以创建具有各种特效和动态效果的图表，例如**数据缩放、鼠标悬停显示数据、动态更新**等。而matplotlib则更加注重于绘图的**基本功能和静态图形的生成**。
4. 应用场景和用户群体：由于其易用性和丰富的可视化效果，Pyecharts通常适用于Web应用程序和数据可视化需求较为复杂的场景，尤其是需要




