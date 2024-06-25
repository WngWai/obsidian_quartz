`pandas`中的`MultiIndex`**多级索引**对象允许在一个轴上拥有多个层级索引。多级索引可以用于**高维数据的重塑和分组**操作

创建多级索引的方法有很多种，其中最常见的方法是**使用层次化索引元组**，或使用`MultiIndex.from_` 系列的工厂函数。

### 创建：
1，直接用二维表：
```python
# 额外添加行列标签名称：
df.index.names=
df.columns.names=
```
![[Pasted image 20240220150037.png]]

2，用pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
![[Pasted image 20240220150201.png]]

```python
import pandas as pd

# 创建多级索引
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

# 创建数据
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8],
                   'B': [10, 20, 30, 40, 50, 60, 70, 80],
                   'C': [100, 200, 300, 400, 500, 600, 700, 800],
                   'D': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]}, index=index)
print(df)
```

输出结果如下：

```python
              A   B    C     D
first second                 
bar   one     1  10  100  1000
      two     2  20  200  2000
baz   one     3  30  300  3000
      two     4  40  400  4000
foo   one     5  50  500  5000
      two     6  60  600  6000
qux   one     7  70  700  7000
      two     8  80  800  8000
```

我们可以看到，这个表中的每一行都有一个二级索引，称为第一层级和第二层级索引。通过固定第一层级的索引，可以很方便地提取具有相同一级索引的所有行。

例如，我们可以使用以下代码提取一级索引为“bar”的所有行。

```python
print(df.loc['bar'])
```

输出结果如下：

```python
        A   B    C     D
second                 
one     1  10  100  1000
two     2  20  200  2000
```



### 多级索引的取值：
[多级索引-知乎](https://zhuanlan.zhihu.com/p/74461994)
[多级索引-B站]([17 怎样使用Pandas的分层索引MultiIndex_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV16R4y147aX/?share_source=copy_web&vd_source=6ba5a7ec009a0f45fd393fcd989921f7))

df.loc[:, [('开户行', ''), ('交易方户名', ''), ('交易卡号', ''), ('收付标志', ''), ('交易金额', 'count'),  ('交易金额', 'sum'), ('反馈结果', '')]]
可以通过**df.index**查看索引结构，针对多级索引，如果有些只有一级，就需要用' '空值来表示，多数以**元组**的形式呈现，用的是**df.loc**进行切片
#### sr
##### 取**某个层级**：
sr['index1'] ，sr['index2'] 
![[Pasted image 20240220160850.png]]


##### 取多个层级：
sr["index1", "index2"] 
sr["index1"]\["index2"\]
![[Pasted image 20240220160859.png]]


##### **切片**：
sr[:, "index2"]
sr.loc[:, "index2"]
![[Pasted image 20240220161301.png]]

#### df
![[Pasted image 20240220164033.png]]

##### 取**某个层级**：
df["column1"] 不推荐
df.loc["column1", :]
![[Pasted image 20240220164046.png]]

![[Pasted image 20240220164127.png]]


##### 取多个层级：
df["column1"]\["column2"\] 不推荐
df.loc[("column1", "column2"), :] **维度和层级**！ 用**元组**取多个层级
![[Pasted image 20240220164111.png]]

![[Pasted image 20240220164143.png]]
![[Pasted image 20240220164201.png]]

##### **切片**：
df.loc\[\[\('index1', 'index1_1'), ('index2', 'index2_2')]\, :] 用**元组列表**取多个多级索引内容！
![[Pasted image 20240220164213.png]]