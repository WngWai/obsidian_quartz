`plt.subplots()` 是 Matplotlib 库中的一个函数，用于创建一个新的 **Figure 对象**并返回一个包含该 Figure 对象及其**所有 Axes 对象的元组**。
```python
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)
```

- `nrows`：表示要创建的子图的行数；
- `ncols`：表示要创建的子图的列数；
- `figsize`：表示**要创建的 Figure 对象的尺寸**，是**元组形式**
- `sharex`：表示**是否共享** X 轴；
- `sharey`：表示是否共享 Y 轴；
- `subplot_kw`：表示要传**递给每个子图的关键字参数**；
- `gridspec_kw`：表示要传递给 `GridSpec` 对象的**关键字参数**。

下面是一个简单的例子，演示了如何使用 `plt.subplots()` 函数创建一个 2x2 的子图布局：
一般用fig(figure), axs(axes)来接

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建一个 2x2 的子图布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

# 在第一个子图中绘制正弦函数
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
axes[0, 0].plot(x, y)
axes[0, 0].set_title('sin(x)')

# 在第二个子图中绘制余弦函数
y = np.cos(x)
axes[0, 1].plot(x, y)
axes[0, 1].set_title('cos(x)')

# 在第三个子图中绘制正切函数
y = np.tan(x)
axes[1, 0].plot(x, y)
axes[1, 0].set_title('tan(x)')

# 在第四个子图中绘制正切函数的导数
y = 1 / np.cos(x) ** 2
axes[1, 1].plot(x, y)
axes[1, 1].set_title('sec^2(x)')

# 设置整个图的标题
fig.suptitle('Trigonometric Functions')

# 显示图形
plt.show()
```

在这个例子中，我们首先使用 `plt.subplots()` 函数创建了一个 2x2 的子图布局，然后在每个子图中绘制了不同的三角函数。注意到 `plt.subplots()` 函数返回的是一个包含 Figure 对象和 Axes 对象的元组，我们可以使用 `axes` 变量来访问每个子图的 Axes 对象。最后，我们还设置了整个图的标题，并调用 `plt.show()` 函数显示图形。

### 子图布局的另一种更随意的方式
逐步创建子图，并进行绘制。但要保证前后一致！避免布置混乱！

通过一个三位数字来指定子图的**行数、列数和索引位置**，是`plt.subplot(nrows, ncols, index)`的简写形式
```python
import matplotlib.pyplot as plt

names = ['A', 'B', 'C']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131) # 1行3列，当前绘制第1个子图
plt.bar(names, values)
plt.subplot(132) # 1行3列，当前绘制第2个子图
plt.scatter(names, values)
plt.subplot(133) # 1行3列，当前绘制第3个子图
plt.plot(names, values)

plt.suptitle('Categorical Plotting') # 添加总标题
plt.tight_layout()
plt.show()
```


### 如何让绘图区不等比例显示
是的，`plt.subplots()`函数也可以实现类似的功能。可以使用`gridspec_kw`参数来传递`GridSpec`对象来创建任意数量和形状的子图。下面是一个示例代码，可以实现将绘图区域划分为1行3列的网格，第一行有两个子图，第二行有一个子图的布局：

代码显示的有点问题？暂时弄不明白，后面再来纠正
```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 4),
                        gridspec_kw={'width_ratios': [2, 1, 1]})

axs[0, 0].plot([1, 2, 3], [4, 5, 6])
axs[0, 0].set_title('subplot 1')

axs[0, 1].plot([1, 2, 3], [6, 5, 4])
axs[0, 1].set_title('subplot 2')

axs[1, :].plot([1, 2, 3], [2, 1, 3])
axs[1, :].set_title('subplot 3')

plt.tight_layout()
plt.show()
```

在这个例子中，我们首先使用`plt.subplots()`函数创建了一个包含2行3列子图的`figure`对象，并将`gridspec_kw`参数设置为一个字典，其中包含一个`width_ratios`键，其值为一个列表，表示每一列的宽度比例。然后，我们使用索引来访问每个子图，并在其上进行绘图和设置标题。最后，使用`tight_layout`方法调整子图的布局，使其在整个绘图区域内居中显示。

`gridspec_kw`是一个关键字参数，用于向`plt.subplots()`函数传递`GridSpec`对象的参数。在这个例子中，我们使用`width_ratios`参数指定了每一列的宽度比例，即第一列的宽度是第二列和第三列的两倍。这个参数的值是一个列表，列表中的每个元素表示相应列的宽度比例。

例如，`gridspec_kw={'width_ratios': [2, 1, 1]}`表示第一列的宽度比第二列和第三列的宽度都要大两倍。因此，在这个例子中，第一列的宽度占整个绘图区域的2/4，第二列和第三列的宽度各占1/4。这样就可以创建一个包含不同大小的子图的网格布局。


