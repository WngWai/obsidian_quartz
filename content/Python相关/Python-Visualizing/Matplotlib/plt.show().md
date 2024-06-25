`plt.show()`是matplotlib库中的一个函数，用于显示绘制的图形。

在使用matplotlib绘图时，通过调用不同的绘图函数（例如`plt.plot()`、`plt.scatter()`等），我们可以生成想要的图形。但是在图形绘制完成后，需要使用`plt.show()`函数来显示图形。

**语法：**
```python
plt.show()
```

**参数说明：**
该函数没有参数。

**注意事项：**
- 在Jupyter Notebook或IPython环境中，可以省略`plt.show()`函数，图形会自动显示。
- 在使用IDE或命令行中使用matplotlib时，必须使用`plt.show()`函数才能显示图形。

**示例：**
下面是一个简单的示例，演示如何使用`plt.plot()`和`plt.show()`绘制并显示折线图。

```python
import matplotlib.pyplot as plt

# 创建示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制折线图
plt.plot(x, y)

# 显示图形
plt.show()
```

运行以上代码，会弹出一个新窗口显示绘制的折线图。