`plt.bar()`是Matplotlib库中的函数，用于绘制条形图（bar chart）。该函数可以用于在图表中创建垂直的条形，每个条形的高度表示相应的值。

函数的语法格式为：

```python
plt.bar(x, height, width=0.8, bottom=None, align='center', data=None, **kwargs)
```

其中，常用参数的含义如下：

- `x`：条形图的x坐标位置。
- `height`：条形图的高度。 就是**纵坐标**
- `width`：条形的宽度，默认值是0.8。
- `bottom`：条形的起始位置，默认值是None。
- `align`：条形的**对齐方式**，默认值是'center'。
- `data`：用于绘制条形图的数据，可以是列表、数组等。
- tick_label 柱状图下的标签
- `**kwargs`：其他可选参数，如颜色、标签等。
1. `color`：设置条形的颜色。可以使用常见的颜色名（如`'red'`、`'blue'`）或颜色代码（如`'#FF0000'`表示红色）。默认值是`None`，表示使用默认颜色。
2. `edgecolor`：设置条形的边框颜色。同样，可以使用颜色名或颜色代码。默认值是`None`，表示使用默认颜色。
3. `label`：为条形图设置标签。可以使用字符串指定标签的内容。通过调用`plt.legend()`函数可以显示标签。默认值是`None`，表示不显示标签。
4. `alpha`：设置条形的透明度。可以传入一个[0, 1]范围内的值，0表示完全透明，1表示完全不透明。默认值是`None`，表示使用默认透明度。
5. `hatch`：设置条形的**填充图案**，可以使用不同的字符来表示不同的图案，如`'/'`、`'\\'`、`'x'`等。默认值是`None`，表示无填充图案。
6. `align`：设置条形的对齐方式。可以设置为`'center'`（默认值，条形在x坐标上居中对齐）、`'edge'`（条形的左边缘与x坐标对齐）或`'edge'`（条形的右边缘与x坐标对齐）。


```python
import matplotlib.pyplot as plt

x = ['A', 'B', 'C', 'D', 'E']
height = [10, 5, 8, 12, 3]

plt.bar(x, height)

plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')

plt.show()
```

这个示例代码绘制了一个简单的条形图，x轴代表不同的类别，y轴代表相应的值。每个条形的高度由`height`列表中的值决定。通过`plt.xlabel()`、`plt.ylabel()`和`plt.title()`函数给图表添加了标签和标题。最后，使用`plt.show()`显示出图表。

请注意，在使用`plt.bar()`之前，通常需要先导入`matplotlib.pyplot`模块，一般以`plt`为别名。