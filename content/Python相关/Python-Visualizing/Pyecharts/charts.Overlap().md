Overlap()是 pyecharts 中的一个类，它可以**将不同类型的图表叠加在一张图上**，实现图表的**组合和对比**。例如，您可以将柱状图和折线图叠加在一起，展示不同维度的数据。

![[Pasted image 20240221084831.png]]
Overlap() 类的使用方法如下：

- 引入 Overlap 类，`from pyecharts import Overlap`
- 实例化 Overlap 类，`overlap = Overlap()`，可指定 `page_title`, `width`, `height`, `jhost` 参数。
- 使用 `add()` 方法向 overlap 中添加图表，可指定 `xaxis_index`, `yaxis_index`, `is_add_xaxis`, `is_add_yaxis` 参数。
- 使用 `render()` 方法渲染生成 `.html` 文件，或者使用 `render_notebook()` 方法在 Jupyter Notebook 中显示图表。

下面是一些使用 Overlap() 类的示例：

- 折线柱状组合图²

```python
from pyecharts import options as opts
from pyecharts.charts import Bar, Line
from pyecharts.faker import Faker

v1 = [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3]
v2 = [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3]
v3 = [2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2]

bar = (
    Bar()
    .add_xaxis(Faker.months)
    .add_yaxis("蒸发量", v1)
    .add_yaxis("降水量", v2)
    .extend_axis(
        yaxis=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value} °C"),
            interval=5
        )
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="折线柱状组合图"),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} ml")),
    )
)

line = Line().add_xaxis(Faker.months).add_yaxis("平均温度", v3, yaxis_index=1)

bar.overlap(line)
bar.render("折线柱状组合图.html")
bar.render_notebook()
```

- 折线散点组合图²

```python
from pyecharts import options as opts
from pyecharts.charts import Line, Scatter
from pyecharts.faker import Faker

x = Faker.choose()
line = (
    Line()
    .add_xaxis(x)
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(title_opts=opts.TitleOpts(title="折线散点组合图"))
)

scatter = (
    Scatter()
    .add_xaxis(x)
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
)

line.overlap(scatter)
line.render("折线散点组合图.html")
line.render_notebook()
```

