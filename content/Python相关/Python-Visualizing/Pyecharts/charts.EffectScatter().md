在Pyecharts库中，`EffectScatter()`函数用于创建涟漪散点图（Effect Scatter chart）。

**函数定义**：
```python
EffectScatter(init_opts: Union[opts.InitOpts, dict] = <pyecharts.options.global_options.InitOpts object>, **kwargs)
```

**参数**：
以下是`EffectScatter()`函数中常用的参数：

- `init_opts`：可选参数，初始化配置项。可以是`opts.InitOpts`类型的对象或包含初始化配置的字典。

- `**kwargs`：其他可选参数，用于设置涟漪散点图的样式、数据等。

**返回值**：
`EffectScatter()`函数返回一个`EffectScatter`类的实例，用于绘制涟漪散点图。

**示例**：
以下是使用`EffectScatter()`函数绘制涟漪散点图的示例：

```python
from pyecharts import options as opts
from pyecharts.charts import EffectScatter

# 创建涟漪散点图
scatter = EffectScatter()

# 设置涟漪散点图的标题和数据
scatter.set_global_opts(title_opts=opts.TitleOpts(title="Effect Scatter Chart Example"))
scatter.add_xaxis(["A", "B", "C"])
scatter.add_yaxis("Category 1", [10, 20, 30], effect_opts=opts.EffectOpts(scale=10))
scatter.add_yaxis("Category 2", [20, 30, 40], effect_opts=opts.EffectOpts(scale=8))

# 渲染生成HTML文件
scatter.render("effect_scatter_chart.html")
```

在上述示例中，我们首先导入了所需的模块和类。

然后，我们使用`EffectScatter()`函数创建了一个涟漪散点图对象`scatter`。

接着，我们使用`set_global_opts()`方法设置了涟漪散点图的标题为"Effect Scatter Chart Example"。

使用`add_xaxis()`方法设置了x轴的刻度标签，这里设置为["A", "B", "C"]。

使用`add_yaxis()`方法添加了两个分类的数据到涟漪散点图。每个分类的数据是一个一维列表，表示每个散点的y轴值。在示例中，我们创建了两个分类的数据，分别是[10, 20, 30]和[20, 30, 40]。

我们还使用`effect_opts`参数设置了涟漪效果的选项，包括`scale`参数来控制涟漪的大小。

最后，我们使用`render()`方法将涟漪散点图渲染为HTML文件。

请注意，上述示例仅演示了基本用法，更多详细的参数和选项可以参考Pyecharts的官方文档。