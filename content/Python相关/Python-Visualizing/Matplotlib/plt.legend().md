是 Matplotlib 库中的一个函数，用于在图表中**添加图例**（legend）。图例是用于标识不同数据系列的符号，通常与图表配合使用，以帮助读者更好地理解数据。

函数签名：
```python
plt.legend(*args, **kwargs)
```

参数说明：
- `*args`：可选参数，用于指定图例中的标签。每个标签对应一个数据系列。
- `**kwargs`：可选参数，用于设置图例的各种属性，如位置、标题、字体大小等。

示例：
```python
import matplotlib.pyplot as plt

# 创建两条曲线
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 8, 27, 64, 125]

plt.plot(x, y1, label='y1 = x^2')
plt.plot(x, y2, label='y2 = x^3')

# 添加图例
plt.legend()

# 显示图表
plt.show()
```

在这个示例中，我们创建了两条曲线 `y1 = x^2` 和 `y2 = x^3`，并使用 `plt.plot()` 函数绘制它们。然后，我们使用 `plt.legend()` 函数添加了图例，使用曲线的标签来标识每个数据系列。

你可以根据需要进一步使用 `**kwargs` 参数来自定义图例的位置、样式和其他属性。例如，可以使用 `loc` 参数指定图例的位置（如 `'upper right'`、`'lower left'` 等），使用 `fontsize` 参数设置图例文本的字体大小等。
