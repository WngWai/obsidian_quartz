`plt.figure()` 是 Matplotlib 库中的一个函数，用于创建一个新**的图形窗口或画布**。

```python
plt.figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
```

- `num`：可选参数，指定图形的编号。如果未指定，将自动分配一个编号。
- `figsize`：可选参数，指定**画布的尺寸**，以英寸为单位的元组 (宽度, 高度)，默认为 **(6.4, 4.8)**
- `dpi`：可选参数，指定画布的**分辨率**（每英寸点数）。默认为 **100**
- `facecolor`：可选参数，指定画布的背景颜色。
- `edgecolor`：可选参数，指定画布的边框颜色。
- `frameon`：可选参数，指定是否绘制画布的边框。默认为 `True`。


```python
import matplotlib.pyplot as plt

# 创建一个新的图形窗口
plt.figure()

# 创建一个编号为 1 的图形窗口并设置画布尺寸
plt.figure(num=1, figsize=(8, 6))

# 创建一个编号为 2 的图形窗口，并设置分辨率和边框颜色
plt.figure(num=2, dpi=200, edgecolor='r')

# 创建一个透明背景的图形窗口
plt.figure(facecolor='none')

# 创建一个没有边框的图形窗口
plt.figure(frameon=False)
```

`plt.figure()` 是创建图形窗口的常用函数，在使用其他绘图函数时，可以使用 `plt.figure()` 创建一个新的图形窗口，并在其中进行绘图操作。

需要注意的是，Matplotlib 允许在不同的图形窗口中同时绘制多张图表，并可以使用 `num` 参数为每个图表指定一个唯一的编号。