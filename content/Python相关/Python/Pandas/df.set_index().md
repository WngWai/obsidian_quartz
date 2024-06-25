在Pandas中，`df.set_index()`函数用于设置DataFrame的索引列。它将指定的列或列组合设置为DataFrame的索引，并返回一个新的DataFrame。
```python
df.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
```
**参数**：
- `keys`：要设置为行引的**列的标签（名称）或标签列表**。可以是单个列名的字符串，也可以是多个列名组成的列表。
- `drop`（可选）：指定是否要**删除设置为索引的列**。默认为True，表示删除。
- `append`（可选）：指定**是否要将新的索引追加**到原有索引之后。默认为False，表示替换原有索引。
- `inplace`（可选）：指定是否在原始DataFrame上进行就地修改。默认为False，表示返回一个新的DataFrame。
- `verify_integrity`（可选）：指定是否要验证新的索引是否唯一。默认为False，表示不验证。

**示例**：
```python
import pandas as pd

# 创建一个示例DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Paris', 'London']}
df = pd.DataFrame(data)

# 将 'Name' 列设置为索引
df_set_index = df.set_index('Name')
print("设置 'Name' 列为索引后的DataFrame：")
print(df_set_index)

# 将多列设置为索引
df_set_multi_index = df.set_index(['Name', 'City'])
print("\n设置多列为索引后的DataFrame：")
print(df_set_multi_index)
```

**输出**：
```python
设置 'Name' 列为索引后的DataFrame：
         Age      City
Name                  
Alice     25  New York
Bob       30     Paris
Charlie   35    London

设置多列为索引后的DataFrame：
                Age
Name    City       
Alice   New York   25
Bob     Paris      30
Charlie London     35
```

在上述示例中，我们首先创建了一个示例的DataFrame `df`，其中包含了三个列：`Name`、`Age`和`City`。然后，我们使用`df.set_index()`函数演示了如何设置指定的列或列组合为DataFrame的索引。

首先，我们调用 `df.set_index('Name')` 将列 `'Name'` 设置为索引。括号中的参数 `'Name'` 是要设置为索引的列的标签。结果存储在 `df_set_index` 中，并打印输出。可以看到，设置 `'Name'` 列为索引后，`df_set_index` 的索引变为了 `'Name'` 列的值。

接下来，我们调用 `df.set_index(['Name', 'City'])` 将多列 `'Name'` 和 `'City'` 设置为索引。括号中的参数 `['Name', 'City']` 是要设置为索引的列的标签列表。结果存储在 `df_set_multi_index` 中，并打印输出。可以看到，设置多列为索引后，`df_set_multi_index` 的索引由 `'Name'` 和 `'City'` 两列组成。

需要注意的是，`df.set_index()`函数默认情况下不会修改原始DataFrame，而是返回一个新的DataFrame。如果想要在原始DataFrame上进行就地修改，可以将 `inplace=True` 作为参数传递给函数。

`df.set_index()`函数在数据操作和数据分析中非常有用，可以将某一列或多列设置为索引，以便更方便地进行数据检索和分析。


### 对于多级索引的情况
只能一次性全部变为列数据，再重新设置1列或多列为行标签！

![[Pasted image 20240220173008.png]]

![[Pasted image 20240220173012.png]]