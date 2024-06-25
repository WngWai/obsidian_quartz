`plt.text()`函数是Matplotlib库中用于在图形中添加文本注释的函数，其语法如下：

```python
plt.text(x, y, s, fontdict=None, withdash=False, **kwargs)
```

其中，参数含义如下：

- `x`：注释文本的横坐标位置；
- `y`：注释文本的纵坐标位置；
- `s`：注释文本的内容；
- `fontdict`：注释文本的字体属性；
- `withdash`：是否使用虚线框将注释文本框起来；
- `**kwargs`：其他参数，包括但不限于文本颜色、文本大小等。

下面是一个例子，展示如何使用`plt.text()`函数添加文本注释：

```python
import matplotlib.pyplot as plt

# 生成数据
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# 绘制散点图
plt.scatter(x, y)

# 添加文本注释
plt.text(2, 10, 'This is a text annotation', fontsize=12, color='red')

# 显示图形
plt.show()
```

在这个例子中，我们使用`plt.scatter()`函数绘制了一个简单的散点图，然后使用`plt.text()`函数在图形中添加了一个文本注释。其中，文本注释的内容为'This is a text annotation'，字体大小为12，颜色为红色，注释文本的位置为(2, 10)。运行代码后，会得到如下的图形：
![[Pasted image 20230605105011.png]]
可以看到，文本注释成功地添加到了图形中，并且位于(2, 10)的位置。

`ha`和`va`参数是用于控制文本注释的水平对齐方式和垂直对齐方式的参数。它们的取值如下：

- `ha`参数：水平对齐方式，可以取值为`'center'`、`'right'`、`'left'`，分别表示文本注释的中心、右侧、左侧与指定位置对齐；
- `va`参数：垂直对齐方式，可以取值为`'center'`、`'top'`、`'bottom'`，分别表示文本注释的中心、顶部、底部与指定位置对齐。

下面是一个例子，展示如何使用`ha`和`va`参数控制文本注释的对齐方式：

```python
import matplotlib.pyplot as plt

# 生成数据
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# 绘制散点图
plt.scatter(x, y)

# 添加文本注释
plt.text(2, 10, 'This is a text annotation', fontsize=12, color='red', ha='center', va='bottom')

# 显示图形
plt.show()
```

在这个例子中，我们将`ha`参数设置为`'center'`，将`va`参数设置为`'bottom'`，这意味着文本注释的中心将与指定位置的水平方向对齐，底部将与指定位置的垂直方向对齐。运行代码后，会得到如下的图形：
![[Pasted image 20230605105513.png]]

可以看到，文本注释的中心与指定位置的水平方向对齐，底部与指定位置的垂直方向对齐。