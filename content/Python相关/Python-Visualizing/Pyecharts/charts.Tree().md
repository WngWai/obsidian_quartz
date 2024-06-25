在Pyecharts库中，`Tree()`函数用于创建树形图（Tree chart）。

**函数定义**：
```python
Tree(init_opts: Union[opts.InitOpts, dict] = <pyecharts.options.global_options.InitOpts object>, **kwargs)
```

**参数**：
以下是`Tree()`函数中常用的参数：

- `init_opts`：可选参数，初始化配置项。可以是`opts.InitOpts`类型的对象或包含初始化配置的字典。

- `**kwargs`：其他可选参数，用于设置树形图的样式、数据等。

**返回值**：
`Tree()`函数返回一个`Tree`类的实例，用于绘制树形图。

**示例**：
以下是使用`Tree()`函数绘制树形图的示例：

```python
from pyecharts import options as opts
from pyecharts.charts import Tree

# 创建树形图
tree = Tree()

# 设置树形图的标题和数据
tree.set_global_opts(title_opts=opts.TitleOpts(title="Tree Chart Example"))
tree.add("", [["A", "B"], ["A", "C"], ["B", "D"], ["B", "E"]])

# 渲染生成HTML文件
tree.render("tree_chart.html")
```

在上述示例中，我们首先导入了所需的模块和类。

然后，我们使用`Tree()`函数创建了一个树形图对象`tree`。

接着，我们使用`set_global_opts()`方法设置了树形图的标题为"Tree Chart Example"。

使用`add()`方法添加了一些数据到树形图。每个数据项都表示一个节点和它的父节点。在示例中，我们创建了一个简单的树形结构，包含了节点"A"、"B"、"C"、"D"和"E"。

最后，我们使用`render()`方法将树形图渲染为HTML文件。

请注意，上述示例仅演示了基本用法，更多详细的参数和选项可以参考Pyecharts的官方文档。