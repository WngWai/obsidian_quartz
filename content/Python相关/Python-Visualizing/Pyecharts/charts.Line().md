在Python的Pyecharts库中，`Line()`函数用于创建折线图（Line Chart）。

```python
import pyecharts.charts as Eplot

line = Eplot.Line()

# 设置图表标题和大小
line_chart.set_global_opts(
    title_opts=opts.TitleOpts(title="折线图示例"),
    visualmap_opts=opts.VisualMapOpts(),
    legend_opts=opts.LegendOpts(is_show=True),
)

# 添加数据
line_chart.add_xaxis(["A", "B", "C", "D", "E"])
line_chart.add_yaxis("系列1", [10, 20, 30, 40, 50])
line_chart.add_yaxis("系列2", [50, 40, 30, 20, 10])

# 将渲染生成的html内容打印出来???
html_text = line.render()
print(html_text)

# 将html内容保存到html文件中
line.render("line.html")
```

**函数定义**：
```python
pyecharts.charts.Line()
```

**参数**：
以下是`Line()`函数中常用的参数：

- `init_opts`（可选）：初始化选项，用于配置图表的基本属性，如标题、背景颜色等。默认值为`opts.InitOpts()`。

- `page_title`（可选）：图表所在HTML页面的标题。默认值为`"Line Chart"`。

- `width`（可选）：图表的宽度。可以是像素值（如`"800px"`）或百分比（如`"80%"`）。默认值为`"800px"`。

- `height`（可选）：图表的高度。可以是像素值（如`"600px"`）或百分比（如`"60%"`）。默认值为`"600px"`。

- `Line()`：创建一个折线图对象，可以使用`add_yaxis()`方法添加一个或多个系列的数据，也可以使用`is_smooth()`方法将折线变为**平滑**曲线，或者使用`is_step()`方法将折线变为 **阶梯**线。

