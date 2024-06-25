sns.pairplot()函数是用于绘制数据集中每对变量之间关系的函数。它可以显示不同变量的联合分布和边际分布，并使用散点图，直方图，核密度估计图，回归图等方式来可视化它们之间的关系。
```python
sns.pairplot()
```

- data：指定输入的数据集，可以是pandas的DataFrame，numpy的数组，或者字典等。
- hue：指定用于分类的变量，可以是数据集中的列名，也可以是向量。
- hue_order：指定hue变量的类别顺序，以列表的形式传递。
- palette：指定用于分类的颜色，可以是字典，列表，或者seaborn支持的颜色格式。
- vars：指定用于绘图的变量，以列表的形式传递，如果不指定，则默认使用数据集中的所有数值变量。
- x_vars, y_vars：指定用于绘图的x轴和y轴的变量，以列表的形式传递，可以与vars参数配合使用，实现更灵活的子图布局。
- kind：指定非对角线上的图的类型，可以是"scatter"（散点图），"kde"（核密度估计图），"hist"（直方图），"reg"（回归图），或者"resid"（残差图）。
- diag_kind：指定对角线上的图的类型，可以是"hist"（直方图），"kde"（核密度估计图），或者None（不绘制）。
- markers：指定散点图的样式，可以是字符串，列表，或者字典。
- height：指定图形的大小（正方形）。
- aspect：指定图形的宽高比例。
- dropna：指定是否删除缺失值。
- plot_kws, diag_kws：指定非对角线上和对角线上的图的额外参数，以字典的形式传递。
- grid_kws：指定轴网格的额外参数，以字典的形式传递。
- 其他参数：根据绘图种类的不同，还可以传递其他参数，例如corner（指定是否只显示左下角的图），levels（指定核密度估计图的等高线数量）等。

sns.pairplot()函数的返回值是一个PairGrid对象，它管理着多个子图，分别对应不同变量的组合。可以通过返回的对象进行更多的自定义操作，例如添加标题，图例，注释等。

下面是一些sns.pairplot()函数的应用举例，使用的数据集是seaborn自带的tips（小费）数据集和iris（鸢尾花）数据集。

```python
# 导入库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# 绘制默认的pairplot
sns.pairplot(tips)
plt.show()

# 绘制分类的pairplot
sns.pairplot(tips, hue="day")
plt.show()

# 绘制回归图和核密度估计图
sns.pairplot(tips, kind="reg", diag_kind="kde")
plt.show()

# 绘制鸢尾花数据的pairplot
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
plt.show()

# 绘制自定义的pairplot
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"], height=4, aspect=1.5, kind="reg")
plt.show()
```
