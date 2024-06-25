`sns.histplot()` 是 Seaborn 库中用于绘制直方图的函数。直方图是统计学中最常见的图形表示之一，用于展示数据的分布情况。该函数提供了丰富的接口用于控制直方图的绘制方式，包括直方图的堆叠、分层等。
```python
seaborn.histplot(_data=None_, _*_, _x=None_, _y=None_, _hue=None_, _weights=None_, _stat='count'_, _bins='auto'_, _binwidth=None_, _binrange=None_, _discrete=None_, _cumulative=False_, _common_bins=True_, _common_norm=True_, _multiple='layer'_, _element='bars'_, _fill=True_, _shrink=1_, _kde=False_, _kde_kws=None_, _line_kws=None_, _thresh=0_, _pthresh=None_, _pmax=None_, _cbar=False_, _cbar_ax=None_, _cbar_kws=None_, _palette=None_, _hue_order=None_, _hue_norm=None_, _color=None_, _log_scale=None_, _legend=True_, _ax=None_, _**kwargs_)
```
### 常用参数介绍：
**data**: 可以是 DataFrame、数组、或列表，指定输入数据。

**x**, **y**: 指定数据中的变量，用于绘制直方图的水平和垂直轴。注意，通常只设置其中之一。

**hue**: 用于分类变量，根**据不同的类别将数据分组**，并使用不同的颜色表示。


**bins**: 指定直方图的**箱数**（简单理解为柱子的数量），可以是整数、**箱边界的序列**，或者是自动计算策略（如 'auto'）。
（1）整数：箱数；
（2）列表：箱子的边界
sns.histplot(tips, x="size", bins=[1, 2, 3, 4, 5, 6, 7])
![[Pasted image 20240423191640.png|400]]

**binwidth**: 指定**每个箱子的宽度**。

**stat**: 统计函数或映射，'count' 表示计数，'frequency' 表示频数，'density' 表示密度（箱面级之和为1），'probability' 表示概率（箱高之和为1）。

？？？row, col：指定用于**分面的分类变量**，可以是数据集中的列名，也可以是向量。这些参数控制用于显示不同子集的子图的布局。

	row分行
	col分列


4. **weights**: 为每个观测值分配权重。

4. **kde**: 布尔值，是否在直方图上叠加**核密度估计**（KDE）。


5. **color**: 指定直方图的颜色。

**binrange**: 指定直方图的范围，即最小和最大边界。
**multiple**: 控制不同层级的直方图如何显示，包括 'layer'、'dodge'、'stack'、'fill' 等。

### 应用举例：

假设你有一个包含人员年龄的 DataFrame `df`，你想要绘制年龄的直方图来查看其分布情况，同时想通过性别来分层显示。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 假设的数据
df = sns.load_dataset('titanic') # 使用 seaborn 内置的 titanic 数据集作为示例

# 绘制直方图
sns.histplot(data=df, x="age", hue="sex", multiple="stack", bins=20, palette="pastel", edgecolor=".3", linewidth=.5)

plt.xlabel('Age') # 设置 x 轴标签
plt.ylabel('Count') # 设置 y 轴标签
plt.title('Age Distribution by Sex') # 设置图表标题
plt.show()
```

在这个例子中，我们使用了 `sns.histplot()` 函数来绘制了一个根据性别分层的年龄分布直方图。通过 `hue` 参数使不同性别的数据以不同颜色显示，并通过 `multiple="stack"` 参数将不同性别的数据堆叠展示，以易于比较不同性别的年龄分布情况。