## 常用快捷键
**Ctrl+q** 在pycharm中查看模块定义

**Ctrl+b** 查看函数、类、方法的定义
## pandas库中的默认值调整
[[pd.set_option()]]

[[Series属性]] **相当于一维数组**
[[DataFrame属性]]


## 数据加载和导出
### Excel
（1）导入
[[pd.read_excel()]] Excel的导入

（2）导出
[[ExcelWriter 对象]] 
[[df.to_excel()]] Excel的写入
[[excel合并]]
[[pd.ExcelWriter()]]
### CSV/TSV
（1）导入
[[pd.read_csv()]] 
TSV类似，只是分隔符需要修改下

（2）导出
[[df.to_csv()]]

### SQL
[[create_engine()]] 创建数据库引擎

[[pd.read_sql()]]直接从数据库中导入数据

[[pd.to_sql()]]将数据直接写入数据库中

### HDF5
二进制文件，能压缩节省空间，能跨平台，迁移到hadoop上。理解为存储三维数据的文件
[[pd.read_hdf()]]

[[df.to_hd()]] 需要指定数据集的“键名”

### JSON
文件内容类似字典，键值为列索引，值为列数据，用到时再好好研究下
[[pd.read_json()]] 

[[df.to_json()]] 

## 数据清洗
### 预览
[[df.info()]] 查看数据基本结构
df.head()
df.tail()

[[df.describe()]]查看数据统计分布情况
### 常用的看和改
（1） 数据类型转换
[[df.dtypes]]  查看df中某列数据的类型
[[sr.dtype]] 看具体某列的数据类型
[[df.astype()]]  转换数据的类型


（2）查漏补缺
[[df.isna()]] 、[[df.notna()]] **查看缺失值**形式更为常见和方便，保持代码的可读性和一致性。这个跟R和SQL还不太一样！主要看null和na的差异！df.isnull()、df.notnull()与之作用相同，用前者。

[[df.dropna()]]**删除缺失值**(NAN)

[[df.fillna()]]**填充缺失值**(NAN)

（3）去重
[[df.duplicated()]] 根据**指定列，标记重复行**
[[df.drop_duplicates()]]根据**指定列，删除重复行**


（4）重新排序
[[df.sort_values()]] 对**内容**进行排序

[[df.sort_index()]] 对**标签**进行排序

[[df.reindex()]] 重新构建**索引顺序**，也可新建**新行、列数据**

### ⭐时间相关
[[Datetime类型]] datetime是内置包，但要调用datetiem模块中的datetime类
[[Timestamp时间戳类对象]] 

新建和转换：
dt=datetime(2023, 11, 2, 1, 30, 45)，得到一个**2023-11-02 01:30:45**。参数是逐级识别的，超出当前日期时间的限定范围会报错。(2023, 11, 2, 1) **2023-11-02 01:00:00**
dt=datetime.now() 获得**当前的日期和时间**，格式同上

[[str=dt.strftime()]]改形式，将datetime对象按**指定的形式输出**

[[dt=df.to_datetime()]] 将**字符串**转为Datatime数据类型

运算：
dt ± [[timedelta()]]

[[Timedelta类对象]] 时间差对象
[[dt.days]] 时间差对象的天数


比较大小：
字符串是文本序列，不能与datetime对象比较大小，可将字符串转换下？？？必须要转换下吗？好像不转换也行，只要字符串的格式较标准。

df['时间'] = df[df['时间'] > '2024-01-02']
'2024-01-02'的效果等同'2024-01-02 00:00:00'

[[df.query()]] 对datetime对象有效，筛选**指定时间**内的内容，**接收字符串的条件表达式**
[[df.beween()]] 对datetime对象也有效，筛选**指定时间**内的内容

[[df.isin()]]结合[[pd.date_range()]] 生成DT格式的时间列表，继续时间匹配查询，只使用短时间内的查询，建议使用上面的查询方式

报错：
ValueError传递**错误的参数**给datetime函数时，如dt = datetime(2021, 11, 1, 25, 0)，超出小时范围
TypeError当尝试将**非数字类型**传递给datetime函数时，如 datetime('2021', 11, 1, 12, 0)

### 新增行列
[[DataFrame属性]] 详见对应数据结构内容

df['new_column']，常结合[[df.apply()]]使用

[[df.reindex()]] 重新排布索引顺序，也可新建新行、列数据。新增行或列可以设置填充对象

### 分组与聚合
实现数据按照指定要求进行分组显示，`要多多多的研究下`
[[df.groupby() 返回SR或DF]]  指定分组索引和聚合函数
[[df.reset_index()]] 将行索引标签释放出来，在分组后就是**释放分组标签**

[[df.apply()]]对df或sr中的**每个元素**应用指定函数
[[df.applymap()]]???

### 交叉表和透视表
根据需求，进行数据汇总
[[pd.crosstab()]] 交叉表，主要是二维变量，主要计算分组的**数量或频数**！！！
[[df.pivot_table()]] 透视表，是分组与聚合的高级应用

### 数据离散化
实现列数据进行重新分组，并扩展到多个列索引中
[[pd.qcut()]] 指定组数
[[pd.cut()]] 自定义分组，一维数组还是一维数组，series还是series
[[pd.get_dummies()]] 的将重新分组后的series离散化为one-hot格式

### 多表连接
[[pd.merge()]]

## DF、SR运算
#### 算术运算
series1 * series2 就能实现对应行数据的相乘，
[[Series运算]]

## 数据分析
#### 统计

- 描述统计

	[[df.describe()]] 生成数值列的基本统计摘要，包括计数、均值、标准差、最小值、25%分位数、中位数、75%分位数和最大值。

	[[df.quantile()]]  看指定分位数对应的值

	- 常用统计方法
		
		count()：计算非缺失值的数量。
		sum()：计算数值的总和。
		mean()：计算数值的平均值。
		median()：计算数值的中位数。
		min()：计算数值的最小值。
		max()：计算数值的最大值。
		std()：计算数值的标准差。
		var()：计算数值的方差。
		cov()：计算两个数值列之间的协方差。
		corr()：计算两个数值列之间的相关系数。

	- 针对pandas的统计

		[[sr.value_counts()]] 统计唯一值出现的频率，不限于数值列，字符串列也行



- 推断统计



## 数据可视化
对于df的可视化，特别是“多绘图区”的显示
[[df.plot()]]
[[plt.show()]]
