`plt.pie()`是matplotlib库中用于绘制饼图的函数，它将数据转换为相对大小的扇形，用于展示各部分数据的占比情况。下面是`plt.pie()`的详细介绍和参数介绍：

```python
plt.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None, radius=None, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False, *, normalize=None, data=None)
```

参数说明：

- `x`：需要绘制饼图的数据，可以是一个数组或者一个序列，每个元素代表一个扇形的大小。

- `explode`：指定每个扇形距离饼图中心的偏移量，用于突出某个扇形，默认为None，即不偏移。

- `labels`：指定**每个扇形的标签**，可以是一个数组或者一个序列，长度必须与`x`相同。

- `colors`：指定每个扇形的颜色，可以是一个数组或者一个序列，长度必须与`x`相同。
- `autopct`：指定每个扇形内部显示的数据格式，可以是一个字符串，也可以是一个函数。
- `pctdistance`：指定每个扇形内部数据标签与圆心的距离。
- `shadow`：是否添加阴影效果。
- `labeldistance`：指定每个扇形标签与圆心的距离。
- `startangle`：指定饼图的起始角度，默认为0度，即从圆心正右侧开始绘制。
- `radius`：指定饼图的半径，默认为1。
- `counterclock`：指定饼图的绘制方向，True表示逆时针绘制，False表示顺时针绘制。
- `wedgeprops`：指定扇形的属性，如**边框线宽、边框颜色**等。
-wedgeprops={'linewidth':0.5, 'edgecolor':'green'}
- `textprops`：指定扇形内部标签的属性，如**字体大小、字体颜色**等。
-textprops={'fontsize':30, 'color':'#003371'}
- `center`：指定饼图的中心点坐标，默认为(0, 0)。
- `frame`：是否显示饼图的边框。
- `rotatelabels`：是否旋转每个标签的角度。
- `normalize`：指定是否对数据进行归一化，使总和为1。
- `data`：指定绘制饼图所在的DataFrame或Series对象。

下面是一个简单的例子，假设我们有一个数组`data`，表示5个城市的人口数量：

```python
import matplotlib.pyplot as plt

data = [12345, 23456, 34567, 45678, 56789]
labels = ['City A', 'City B', 'City C', 'City D', 'City E']
colors = ['r', 'g', 'b', 'c', 'm']

plt.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()
```

这段代码会绘制一个饼图，每个扇形代表一个城市的人口数量，标签为城市名称，颜色为红、绿、蓝、青、洋红五种颜色，内部显示数据的格式为百分数，添加阴影效果，起始角度为90度。