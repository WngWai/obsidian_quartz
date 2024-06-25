sns.distplot()函数是用于绘制单变量分布的函数，它可以同时显示直方图和核密度估计曲线，以及观测值的小细条和参数分布的拟合曲线¹。

看GPT4.0说是distplot()被弃用了，现在主要被histplot()和kdeplot()取代了。

```python
sns.distplot()
```

- a：指定**输入的数据**，可以是一维数组，Series，DataFrame列，numpy数组等。

- bins：指定直方图的划分，可以是**整数**，**列表**，或者字符串（如"auto"，"sturges"，"doane"等）。
- binwidth **箱宽**

- hist：指定**是否显示直方图**，布尔值，默认为True。


- row, col：指定用于**分面的分类变量**，可以是数据集中的列名，也可以是向量。这些参数控制用于显示不同子集的子图的布局。

	row分行
	col分列



待整理：

- kde：指定是否显示核密度估计曲线，布尔值，默认为True。
- rug：指定是否显示观测值的小细条，布尔值，默认为False。
- fit：指定用于拟合参数分布的分布类，例如scipy.stats中的norm，gamma，lognorm等。
- hist_kws, kde_kws, rug_kws, fit_kws：指定直方图，核密度估计曲线，小细条，拟合曲线的额外参数，以字典的形式传递。
- color：指定绘图的颜色，可以是matplotlib支持的颜色格式。
- vertical：指定是否将图形垂直显示，布尔值，默认为False。
- norm_hist：指定是否将直方图归一化为密度，布尔值，默认为False。
- axlabel：指定x轴的标签，字符串，默认为None。
- label：指定图例的标签，字符串，默认为None。
- ax：指定绘图的轴，matplotlib的Axes对象，默认为None。

sns.distplot()函数的返回值是一个Axes对象，它包含了绘制的图形，可以进行更多的自定义操作，例如添加标题，图例，注释等。

下面是一些sns.distplot()函数的应用举例，使用的数据是numpy生成的服从正态分布的随机数。

```python
# 导入库
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
x = np.random.randn(100)

# 绘制默认的distplot
sns.distplot(x)
plt.show()

# 绘制不显示核密度估计曲线的distplot
sns.distplot(x, kde=False)
plt.show()

# 绘制不显示直方图的distplot
sns.distplot(x, hist=False)
plt.show()

# 绘制显示观测值小细条的distplot
sns.distplot(x, rug=True)
plt.show()

# 绘制拟合正态分布的distplot
from scipy.stats import norm
sns.distplot(x, fit=norm)
plt.show()

# 绘制自定义颜色和样式的distplot
sns.distplot(x, color="red", hist_kws={"alpha":0.5, "edgecolor":"black"}, kde_kws={"linewidth":3, "linestyle":"dashed"})
plt.show()
```
