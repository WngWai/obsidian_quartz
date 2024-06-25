[[pd.DataFrame()]]
[[df.to_numpy()]] 将df数据结构转换为numpy数组

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
s = df['A']  # s的索引是df的索引

# 输出series
print(s)
0    1
1    2
2    3
Name: A, dtype: int64

print(s.dtype)
int64 

# 输出df
print(df)
   A  B
0  1  a
1  2  b
2  3  c

print(df.dtypes)
A     int64
B    object
dtype: object

```


## 基础属性：
[[df.dtypes]]查看df整体的数据类型 
[[df.shape]]返回先行后列的**元组**
[[df.index]] 反回行标签数据，index索引对象，可以直接当列表使用
[[df.index.names]] 获得行索引名称
[[df.index.levels]] 获得索引层级？
[[df.columns]] 反回列标签数据，index索引对象，可以直接当列表使用
[[df.values]] 返回ndarray

## 索引
df\[[column1, column2...]]\[[index1, index2...]] **直接**索引，先列后行，扩展为单例、布尔。但不能像列表那边直径进行数字索引
df.column1 **很少用**，也是索引df中的某一列，输出sr
[[df.loc]][]行、列标签**名称**索引
[[df.iloc]][]整数**位置**（integer location）索引，在范围索引上也是**含左不含右**

[[MultiIndex多级索引]] 
定义：
选取数据：
在分组聚合中的影响：

涉及到逻辑运算，尽量用&和|，这样不会出错！

## 查：
[[df.query()]] 类似SQL的查询语句，简洁易读
[[df.dtypes]] values中数据格式查询
type(df) 直接看整体的数据格式
df.head() 默认前五行
df.tail() 默认后五行

## 增：
### 合并：
[[pd.merge()]] 按方向拼接，可以实现**多表间的内外连接**
[[pd.concat()]] 按索引拼接

### 添加：
[[df.insert()]] 添加**新列**数据

df.append() 失效
创建df=pd.DataFrame()空白df，添加新行数据是可以的
[[df._append()]]在**df末尾添加新行或其他df**，原始的df可以是空值！

df.loc[len(df)] = new_df **前提是df中必须有数据**，否则为空时识别第一行会报错？

## 删除：
[[df.drop()]] 删除指定行列

## 改：
### 属性
[[df.astype()]] 数据**类型转换**
df_data['交易金额'] = df_data['交易金额'].astype('float') 数据类型转换为浮点数
df_data['交易金额'].dtype = 'float' 这种方法不会改变数据的内容，只是改变Pandas内部存储数据的方式

### 行/列标签
df.index=[...] 修改行标签，无法单独修改某一个标签值，但可以整体修改
df.columns=[...] 修改列标签，无法单独修改某一个标签值，但可以整体修改

[[Python相关/Python/Pandas/rename()|rename()]] 修改**行、列名称**，用新名代替旧名

[[df.reindex()]] **重新排序**行、列标签

[[df.set_index()]]将**某列数据设置为行标签**，可以用于**设置多级索引**
[[df.reset_index()]] 重置索引，在分组聚合后经常用到，将**行索引变为列数据**，新增默认的0、1、2...等默认行索引

### 元素
[[df.T]] 
[[df.sort_index()]]  对**标签名**进行排序
[[df.sort_values()]]  对**内容**进行排序

[[df.shift()]]实现**df数据错一行**，从而计算时间序列上不同时间段的变化值
## 运算
[[广播操作]]
[[可广播的对象]]

### 算术运算
别用**df+3**这种操作了，直接用函数！
元素级别的操作
[[df.add()]]加法
[[df.sub()]]减法
[[df.mul()]]乘法
[[df.div()]]除法
[[df.nunique()]]返回唯一值的数量，int型数据
[[df.unique()]]返回唯一值的元素值


### 逻辑运算
[[df逻辑运算]]
[[df.isin()]]满足要求的数值返回true值，否则为false值，可以用来筛选df数值，相当于df[df['colum1'] == 0]

### 统计运算
[[df.describe()]]查看数据统计分布情况

### 自定义运算
[[df.apply()]] 对字段中的单个元素进行操作

### 迭代
[[df.iterrows()]] 返回每一行行标签和行数据数据的元组
