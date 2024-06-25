在Python的pandas包中，`groupby()`函数用于对数据进行分组操作。它将数据按照指定的**列或多列进行分组**，并提供了对每个组进行**聚合、转换和过滤**的功能。
![[Pasted image 20231213110657.png]]

**函数定义**：

```python
DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, **kwargs)
```

**参数**：

- `by`：用于分组的**列名、列名列表、Series、索引**等。可以是单个列名的字符串，也可以是多个列名组成的列表。
```python
df.groupby('column1')
df.groupby(['column1', 'column2'])  
```

- `axis`：指定分组方向，0表示按行分组，1表示按列分组，默认为**0**，按行分组。

- `level`：如果数据框具有**多层索引**，则可以指定要使用的索引级别进行分组。？？？
```python
df.groupby(level=0)  # 按照第一层级进行分组
df.groupby(level=[0, 1])  # 按照第一层级和第二层级进行分组
```

- `as_index`：指定是否**将分组的列作为索引**，默认为True。如果设置为False，则分组的列将作为普通列保留在数据框中。如果不使用，一般使用[[reset_index()]]
```python
df.groupby('column1', as_index=False)  # 分组后，不将 'column1' 列设置为索引
```

- `sort`：是否按照分组键进行排序，指定是否**对分组的结果**进行排序，默认为True。？？？
```python
df.groupby('销售员姓名', sort=False)  # 分组后，不按照 '销售员姓名' 进行排序
```

- `group_keys`：指定是否显示**分组的键**，默认为True。
```python
df.groupby('column1', group_keys=False)  # 分组后，不在结果中包含 'column1' 组键
```

- `squeeze`：指定是否在可能的情况下**对分组结果进行压缩**，默认为False。如果仅有一组，则返回一个Series。
- `observed`：指定是否仅使用**观察到的分组值**进行分组，默认为False。



**示例**：
```python
df[['col1', 'col2', 'col3', 'col4']]  是在df中选取四列，形成新的df

.groupby(['col1', 'col2']) 对col1，col2列进行逐级分组

['col3'].sum() 对分组后的组内列数据进行聚合操作

[['col3', 'col4']].sum() 对col3，col4列执行相同聚合操作
 
.agg({'col3': sum}, {'col4': mean}) 两组分组按照多个聚合函数集合，输出两组DF
```

![[Pasted image 20240218201741.png]]

这个作者的文件建议全部看下！
https://zhuanlan.zhihu.com/p/101284491

### [[agg()]]聚合函数
![[Pasted image 20231213110904.png]]

[[df.sum()]]


例如sum、mean、max、min等。例如，我们有一个DataFrame对象df，其中包含“Country”和“Sales”两列数据，我们想要按照“Country”列进行分组，并对每个分组的“Sales”列求和、平均值、最大值和最小值，代码如下：

   ```python
   df.groupby('Country')['Sales'].sum()
   ```
这将返回一个Series对象，其中包含按“Country”列分组后**每个分组的“Sales”列求和的结果**。

   ```python
   df.groupby('Country')['Sales'].agg(['sum', 'mean', 'max', 'min'])
   ```
这将返回一个DataFrame对象，其中包含按“Country”列分组后每个分组的“Sales”列求和、平均值、最大值和最小值的结果。

```python
.agg({'order_id':'nunique', 'price':'sum'})
```
分组后对**多个列索引**数据进行聚合，此时**返回的是DF而非SR了**！

#### [[df.unique()]]

对serize数据进行操作，可以将分组后的**唯一值都放入列表中**！
```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
    'City': ['London', 'New York', 'London', 'Paris', 'Paris'],
    'Age': [25, 30, 35, 40, 45],
    'Salary': [5000, 6000, 5500, 7000, 6500]
}
df = pd.DataFrame(data)

unique_cities = df['City'].unique()
print(unique_cities)
# 输出: ['London' 'New York' 'Paris']
```

#### join
```python
import pandas as pd

# 创建一个示例 DataFrame
data = {'ID': [1, 1, 2, 2, 2],
        'Value': ['A', 'B', 'C', 'D', 'E']}
df = pd.DataFrame(data)

# 使用 groupby 和 agg 进行聚合
result = df.groupby('ID')['Value'].agg(lambda x: ', '.join(x)).reset_index()

print(result)

```

### agg()自定义聚合函数

`groupby()`函数还支持应用自定义聚合函数。例如，我们有一个DataFrame对象df，其中包含“Country”和“Sales”两列数据，我们想要按照“Country”列进行分组，并对每个分组的“Sales”列应用自定义聚合函数，代码如下：

   ```python
   def my_agg(x):
       return x.max() - x.min()

   df.groupby('Country')['Sales'].agg(my_agg)
   ```
这将返回一个Series对象，其中包含按“Country”列分组后每个分组的“Sales”列应用自定义聚合函数的结果。



### 聚合列如何指定名称
在Pandas中使用`groupby`进行分组聚合后，默认情况下，聚合列的名称会根据聚合操作而自动确定。例如，如果你对某个列执行了求和（`.sum()`）操作，那么聚合后的列名通常会保留原列名。但是，如果你想要为聚合后的列指定更具体的名称，可以通过多种方式来实现，其中一种常见的方式是使用`.agg()`方法，并传递一个字典来明确指定列名和操作。

假设我们有以下DataFrame：

```python
import pandas as pd

# 创建示例DataFrame
data = {'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Values': [10, 15, 10, 20, 5, 25]}
df = pd.DataFrame(data)

print(df)
```

假设我们想要按`Category`分组，并对`Values`列求和，同时为聚合后的列指定名称`'Total'`：

1，使用`.agg()`方法（推荐）

```python
result = df.groupby('Category')['Values'].agg(Total='sum').reset_index()
print(result)
```

在这个例子中，我们使用`.agg()`方法并传递了一个字典`{'Total': 'sum'}`来指明我们想要对`Values`列执行的操作（`'sum'`）以及聚合后列的新名称（`'Total'`）。

2，使用字典进行聚合
字典的键是我们想要聚合的列名，值是一个元组，其中第一个元素是我们想要的新列名，第二个元素是聚合函数。

另一种方式，如果需要对多个列进行不同的聚合操作并同时重命名这些聚合列，可以这样做：

```python
# 假设我们有另一个列，并希望同时对两个不同的列执行不同的聚合操作
data['AnotherVal'] = [1, 2, 3, 4, 5, 6]
result = df.groupby('Category').agg({'Values': ('Total', 'sum'), 'AnotherVal': ('Average', 'mean')}).reset_index()

# 重置列名称
result.columns = ['Category', 'Total', 'Average']

print(result)
```

在这个例子中，我们传递了一个字典，其中键是待聚合的列名，值是一个元组，元组的第一个元素是我们期望的新列名，第二个元素是聚合函数。

3，直接单列聚合
聚合后的名称就是**原列名**！
```python
df_mac = df_data.groupby('MAC地址')['交易方证件号码'].count().reset_index()

# 输出：
              MAC地址  交易方证件号码
0                  0        1
1  00-00-00-00-00-00        1
2  00-E0-0C-05-03-BA        1
3       00CFE04293B6        1
4       00E04C4853C7        1
```




### [[transform()]]
用于**执行组级别的操作**，并将结果**广播到原始 DataFrame 的相应行上**

![[Pasted image 20231213110322.png]]
**示例：**
```python
import pandas as pd

# 创建一个DataFrame
data = {'Group': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 15, 20, 25, 30]}
df = pd.DataFrame(data)

# 定义一个函数，计算每个组的平均值
def group_mean(x):
    return x.mean()

# 使用 transform 应用 group_mean 函数
df['Group_Mean'] = df.groupby('Group')['Value'].transform(group_mean)

print(df)
```

**输出：**
```
  Group  Value  Group_Mean
0     A     10          20
1     B     15          20
2     A     20          20
3     B     25          20
4     A     30          20
```

### 常见错误
#### 1，分组对象作为列索引，导致索引不到分组列
要么分组时设置为False，要么后面df.reset_index()，释放行标签

#### 2，agg({})对分组后的多个列无法执行聚合操作
只有一个字典内容！不要写成多个字典，除了第一个字典外其他字典内容的列字段执行聚合操作不会被执行！
```python
# 正确
df.groupby(['column']).agg({'order_id':'nunique', 'price':'sum'})

# 错误
df.groupby(['column']).agg({'order_id':'nunique'}, {'price':'sum'})
```