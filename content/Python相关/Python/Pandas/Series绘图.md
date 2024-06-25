### 单绘图区：sr.plot.barh();sr.plot()折线图
**plt.barh(sr)**；**plt.plot(sr)** 也行。
针对DF、SR标准的应该采用**SR.plot(kind="")**

`Series` 对象**没有** `plt` 属性，因此无法使用 sr.plt.barh()这种方式绘制水平条形图。折线图

如果你想使用 `Series` 对象绘制水平条形图，可以使用 `Series` 对象自带的 `plot.barh()` 方法，例如：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建一个Series对象
sr = pd.Series(np.random.rand(5), index=list('abcde'))

# 使用Series的plot方法绘制水平条形图
sr.plot.barh()
plt.show()
```

如果你想使用 `plt.barh()` 方法绘制水平条形图，需要将 `Series` 对象的索引作为 `y` 轴数据，将 `Series` 对象的值作为 `x` 轴数据，例如：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建一个Series对象
sr = pd.Series(np.random.rand(5), index=list('abcde'))

# 使用plt.barh方法绘制水平条形图
plt.barh(sr.index, sr.values)
plt.show()
```

无论你使用哪种方式，都需要导入 matplotlib 库，并在绘制完图形后使用 `plt.show()` 方法显示图形。


### 多个绘图区：sr.plot(ax=axs[?], kind="")
在 Pandas 中，可以使用 `subplots` 参数来实现在多个绘图区中绘制多个折线图。具体步骤如下：

1. 创建一个 `Series` 对象，可以使用 Pandas 的 `Series()` 函数。
2. 创建一个 `matplotlib` 的 `Figure` 对象和多个 `Axes` 对象，可以使用 `plt.subplots()` 函数来创建。
3. 在每个 `Axes` 对象中调用 `Series.plot()` 方法，传入 `kind='line'` 参数即可画出折线图。
4. 可以对每个 `Axes` 对象进行进一步的定制，例如添加标题、轴标签等。
5. 最后使用 `plt.show()` 方法显示图形。

下面是一个示例代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建一个 Series 对象
s = pd.Series(np.random.randn(10).cumsum())

# 创建 Figure 和 Axes 对象
fig, axes = plt.subplots(nrows=2, ncols=1)

# 在第一个 Axes 中画折线图
s.plot(ax=axes[0], kind='line')
axes[0].set_title('First Plot')
axes[0].set_xlabel('X Label')
axes[0].set_ylabel('Y Label')

# 在第二个 Axes 中画折线图
s.plot(ax=axes[1], kind='line')
axes[1].set_title('Second Plot')
axes[1].set_xlabel('X Label')
axes[1].set_ylabel('Y Label')

# 显示图形
plt.show()
```


### 为什么axs[0].plot(data1)没问题
`axs[1].bar(data1)` 会报错，因为 `bar()` 方法需要传入两个参数，即 x 轴和 y 轴的数据。而 `axs[0].plot(data1)` 没有问题，是因为 `plot()` 方法默认将 `data1` **视为 y 轴数据**，x 轴数据**默认为该 Series 的索引**。

如果想要在 `axs[1]` 中画柱状图，**需要同时传入 x 轴和 y 轴的数据**。例如：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建一个 Series 对象
data1 = pd.Series(np.random.randn(10).cumsum())

# 创建 Figure 和 Axes 对象
fig, axs = plt.subplots(nrows=2, ncols=1)

# 在第一个 Axes 中画折线图
axs[0].plot(data1)
axs[0
```


### 如何添加标签
要添加标签，可以使用 Matplotlib 库中的方法。Pandas 的 `plot()` 方法返回的是一个 Matplotlib 的 AxesSubplot 对象，因此可以使用该对象上的方法添加标签。下面是一个示例：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 创建数据
data = pd.DataFrame({'年份': [2015, 2016, 2017, 2018, 2019], '销售额': [100, 120, 150, 180, 200]})

# 绘制折线图
ax = data.plot(x='年份', y='销售额', kind='line')

# 添加标题和标签
ax.set_title('销售额变化趋势')
ax.set_xlabel('年份')
ax.set_ylabel('销售额')

# 显示图形
plt.show()
```

在上述代码中，我们首先使用 `plot()` 方法绘制了折线图，并将返回的 AxesSubplot 对象保存到了 `ax` 变量中。然后，我们使用该对象上的 `set_title()`、`set_xlabel()` 和 `set_ylabel()` 方法分别添加了标题和 x、y 轴的标签。最后，使用 `plt.show()` 方法显示图形。