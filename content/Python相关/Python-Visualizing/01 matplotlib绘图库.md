```python
import matplotlib.pyplot as plt
```

https://zhuanlan.zhihu.com/p/577793239
看是否需要详细添加一些额外说明！
matplotlib 的画图功能是如何实现的？ - AIquant的回答 - 知乎
https://www.zhihu.com/question/559044784/answer/3337222825


[官方文档](https://matplotlib.org/stable/users/explain/quick_start.html)

[数据可视化工具目录](https://datavizcatalogue.com/ZH/)

python包中matplotlib、Pyecharts、Seaborn包的主要区别，这个问题的答案是**这三个包都是用于数据可视化的库，但各有特点和优势**。
- matplotlib是Python数据可视化库中的**泰斗**，它提供了**一整套和MATLAB相似的命令API**，适合交互式制图，也可以将它作为绘图控件，嵌入其它应用程序中。它支持多种图表类型，如折线图，柱状图，饼图，散点图，直方图，箱线图，等高线图，三维图等。它的输出的图片质量也达到了科技论文中的印刷质量，**日常的基本绘图**更不在话下。
- Seaborn是**基于matplotlib**的图形可视化python包，它在matplotlib的基础上进行了**更高级的API封装**，提供了一种**高度交互式**界面，从而使得作图更加容易，便于用户能够做出各种有吸引力的统计图表。它能高度兼容numpy与pandas数据结构以及scipy与statsmodels等统计模式。它支持多种图表类型，如折线图，柱状图，饼图，散点图，直方图，箱线图，小提琴图，条形图，热力图，联合分布图，回归图，分面网格图等。它的图表具有更美观、更现代的调色板设计和样式设置，可以让图表更加清晰和美观。 
- Pyecharts是基于 Echarts 开发的，是一个用于**生成 Echarts 图表**的类库。Echarts 是**百度开源的一个数据可视化 JS 库**，凭借着良好的交互性，精巧的图表设计，得到了众多开发者的认可。而Pyecharts，实际上就是 Echarts 与 Python 的对接。它支持多种图表类型，如折线图，柱状图，饼图，散点图，地图，仪表盘，漏斗图，雷达图，平行坐标图，桑基图，关系图，热力图，水球图，词云图，3D图等。它的图表具有**高度的交互性和动画效果**，可以轻松搭配出精美的视图。

![[Pasted image 20240227130332.png]]
画布
绘图区
绘图
辅助显示层：x、y轴的坐标、显示范围、刻度名称、标题、网格
图例、图形显示

### 容器层
[[plt.figure()]] 创建**画布**

### 图像层
[[plt.plot()]] 绘制折线图或者曲线图
[[plt.scatter()]] 散点图
[[plt.bar()]] 柱状图，统计/对比，多组数据的分布情况，比如**按年分布**
[[plt.hist()]] 直方图，统计**同一类的连续数据**在不同区间上的分布状况，高度表示频数，面积表数量，横轴表统计区间
[[plt.pie()]] 饼图，用的不多

### 辅助显示层
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

plt.xlim()X轴显示的范围
plt.ylim()
[[plt.xticks()]] X轴**刻度名称**，指定刻度对应的**刻度标签**内容！
plt.yticks() 同上
[[plt.xlabel()]] X轴**标签**
plt.ylabel() Y轴标签
[[plt.title()]] **标题**

[[plt.grid()]] **网络**显示
[[plt.legend()]] 显示**图例**，**绘图区的label**才能显示出来
[[plt.show()]] **绘制**图像

[[plt.savefig()]] **保存**图像。在调用plt.show()前用
### 多绘图区
[[Axes对象]]
[[fig, axes = plt.subplots()]] 在画布上定义**多个绘图区**

axes[0].plot() 其他图形同上

axes[0].set_xlim()显示的范围
axes[0].set_ylim()
axes[0].[[set_xticks()]]刻度
axes[0].set_yticks()
axes[0].set_xticklabels()**刻度标签**
axes[0].set_yticklabels()
axes[0].[[set_xlabel()]]标签
axes[0].set_ylabel()
axes[0].set_title() 子绘图区域的标题

axes[0].grid() 在指定的axes对象中显示网格
axes[0].legend() 显示图例

[[fig.suptitle()]] 设置所有子图的总标题

plt.show()

### 关于df的绘图
[[df.plot()]]
