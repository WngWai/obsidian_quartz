用于保存图形的函数。该函数将当前的图形**保存为指定格式**的文件，可以保存为多种格式，如 PNG、PDF、SVG、EPS 等等。

```python
plt.savefig(filename, format=None, dpi=None, bbox_inches=None, pad_inches=None)
```

`filename` 参数指定保存的文件名，可以是绝对路径或相对路径。`format` 参数指定保存的文件格式，可以是 **PNG、PDF、SVG、EPS** 等等。如果不指定该参数，则默认保存为 PNG 格式。

`dpi` 参数指定保存的图像分辨率，即每英寸包含的像素数。

`bbox_inches` 参数指定保存的图像边界框，可以是 'tight'、'standard' 等等。

`pad_inches` 参数指定保存的图像周围的空白区域大小，以英寸为单位。

例如，将当前的图形保存为 PNG 格式的文件：

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.savefig('plot.png')
```

执行以上代码后，将在当前工作目录下保存一个名为 `plot.png` 的文件。

### 为什么保留的图片为空白
在使用Matplotlib绘图时，如果你先调用`plt.show()`再调用`plt.savefig()`保存图像，你得到的图像文件将会是空白的。

这是因为`plt.show()`会导致当前图形被显示后，图形的绘制**缓冲区被清空**（也就是说，图形已经被渲染到屏幕上，然后清空缓冲区以便开始新的图形绘制），因此当你随后调用`plt.savefig()`时，由于绘制缓冲区已经被清空，所以保存的是一个空白的图像文件。

为了解决这个问题，你应该在调用`plt.show()`之前调用`plt.savefig()`。这样做可以确保你保存的图像包含你想要的所有绘图内容。