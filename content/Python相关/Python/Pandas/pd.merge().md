是 Pandas 库中用于连接两个 DataFrame 的方法，它类似于 SQL 中的 join 操作。通过`merge()`函数，我们可以根据一个或多个键（即列）来**合并两个 DataFrame**。

在 Pandas 中，可以使用 `merge()` 函数来实现左连接操作。左连接是根据一个或多个键将两个 DataFrame 进行合并，保留左侧 DataFrame 的所有行，并将右侧 DataFrame 中匹配的行合并到左侧。

```python
merged_df = pd.merge(left_df, right_df, how='left', on='key')
```

其中，`left_df` 是左侧的 DataFrame，`right_df` 是右侧的 DataFrame，`how='left'` 表示进行左连接操作，`on='key'` 表示按照 'key' 列进行连接。你可以根据实际情况修改 'key' 列的名称。

以下是一个示例，展示如何使用左连接合并两个 DataFrame：

```python
import pandas as pd

# 创建左侧 DataFrame
left_df = pd.DataFrame({'A': ['A1', 'A2', 'A3'],
                        'B': ['B1', 'B2', 'B3']})

# 创建右侧 DataFrame
right_df = pd.DataFrame({'A': ['A2', 'A3', 'A4'],
                         'C': ['C2', 'C3', 'C4']})

# 左连接操作
merged_df = pd.merge(left_df, right_df, how='left', on='A')
print(merged_df)
```

输出结果如下：

```
    A   B    C
0  A1  B1  NaN
1  A2  B2   C2
2  A3  B3   C3
```

在上述示例中，左侧 DataFrame `left_df` 包含两列 'A' 和 'B'，右侧 DataFrame `right_df` 包含两列 'A' 和 'C'。通过左连接操作，根据 'A' 列进行合并，并保留左侧 DataFrame 的所有行。右侧 DataFrame 中与左侧匹配的行合并到左侧 DataFrame，没有匹配的行则在 'C' 列中填充 NaN。

你可以根据实际需求进行灵活的列名和连接键的设置，以适应你的数据合并需求。


### 示例1：使用单个键合并两个 DataFrame

假设我们有两个 DataFrame：`df1`和`df2`，如下所示。

```Python
import pandas as pd
df1 = pd.DataFrame({"key": ["A", "B", "C", "D"], "value": [1, 2, 3, 4]})
df2 = pd.DataFrame({"key": ["B", "D", "E", "F"], "value": [5, 6, 7, 8]})
```

现在，我们可以使用`merge()`函数来基于键“key”来合并这两个 DataFrame。

```Python
merged = pd.merge(df1, df2, on="key")
print(merged)
```

输出为：

```python
  key  value_x  value_y
0   B        2        5
1   D        4        6
```

在这个例子中，Pandas 首先将两个 DataFrame 按照“key”列进行匹配，并将它们合并为一个新的 DataFrame。在新的 DataFrame 中，第一个 DataFrame 的值以“_x” 结尾，第二个 DataFrame 的值以“_y”结尾，以便区分它们是来自哪个 DataFrame。

### 示例2：使用多个键合并两个 DataFrame

有时，使用多个键（即多个列）来合并两个 DataFrame 会更加方便。例如，假设我们有以下两个 DataFrame。

```Python
df3 = pd.DataFrame({"key1": ["A", "B", "C", "D"], "key2": ["W", "X", "Y", "Z"], "value": [1, 2, 3, 4]})
df4 = pd.DataFrame({"key1": ["B", "D", "E", "F"], "key2": ["X", "Z", "A", "B"], "value": [5, 6, 7, 8]})
```

可以看到，这两个 DataFrame 都有两个键（即列）：`key1`和`key2`。

现在，我们可以使用两个键来合并这两个 DataFrame。

```Python
merged2 = pd.merge(df3, df4, on=["key1", "key2"])
print(merged2)
```

输出为：

```python
  key1 key2  value_x  value_y
0    B    X        2        5
1    D    Z        4        6
```

在这个合并中，Pandas 会在两个 DataFrame 中找到`key1`、`key2`都匹配的行并进行合并。

这就是`merge()`函数的使用方法和一些示例。Pandas 提供了许多选项，以帮助你根据需要进行合并。关于更高级的选项，请务必查看 Pandas 的官方文档。


在Pandas库中，merge()函数可以完成四种不同方式的数据连接（join）：left join、right join、inner join、outer join。

下面我们逐个来介绍这四种不同方式的数据连接方法，以及它们的使用场景及举例说明：

### Left Join

Left Join（左中所有记录以及右表中已经匹配到的记录。如果在右表中没有找到匹配的，则在结果中用NULL填充。例如：

```Python
import pandas as pd

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

left_join = pd.merge(df1, df2, how='left', on='key')
print(left_join)
```

以上代码执行后，输出结果为：

```python
  key  value1  value2
0   A       1     NaN
1   B       2     5.0
2   C       3     NaN
3   D       4     6.0
```

可以看到，对于df1中的每一行，我们都在df2中找到一行与之匹配。对于df1中的键为‘A’的行，在df2中没有匹配到符合条件的行，因此在右侧添加一个NaN值。

### Right Join

Right Join（右连接），返回右表中所有记录以及左表中已经匹配到的记录。如果在左表中没有找到匹配的，则在结果中用NULL填充。 例如：

```Python
import pandas as pd

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

right_join = pd.merge(df1, df2, how='right', on='key')
print(right_join)
```

以上代码执行后，输出结果为：

```python
  key  value1  value2
0   B     2.0       5
1   D     4.0       6
2   E     NaN       7
3   F     NaN       8
```

可以看到，对于df2中的每一行，我们都在df1中找到一行与之匹配。对于df2中的键为‘E’、‘F’的行，在df1中没有匹配到符合条件的行，因此在左侧添加一个NaN值。

### Inner Join

Inner Join（内连接），返回两个表中都匹配到的部分。例如：

```Python
import pandas as pd

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

inner_join = pd.merge(df1, df2, how='inner', on='key')
执行后，输出结果为：
```

```python
    B       2       5
1   D       4       6
```

可以看出，内连接结果只包括 df1 和 df2 都有的行，没有包括 df1 或 df2 中独有的行。

### Outer Join

Outer Join（外连接）或叫 Full Join（完全连接），返回左右表中所有记录，没有匹配到的位置填充为null。例如：

```Python
import pandas as pd

df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

outer_join = pd.merge(df1, df2, how='outer', on='key')
print(outer_join)
```

以上代码执行后，输出结果为：

```python
  key  value1  value2
0   A     1.0     NaN
1   B     2.0     5.0
2   C     3.0     NaN
3   D     4.0     6.0
4   E     NaN     7.0
5   F     NaN     8.0
```

可以看出，完全连接结果包括 df1 和 df2 两个数据集所有的行，对于某一行在 df1 中找不到对应的行，在输出结果中相应位置填充 null。

### semi_join
![[Pasted image 20231212153957.png]]
交集的左侧
```python
result_semi_join = df_left[df_left['A'].isin(df_right['A'])]
```


### anti_join
![[Pasted image 20231212154007.png]]
交集的左侧的剩余，加了个逻辑非“~”
```python
result_anti_join = df_left[~df_left['A'].isin(df_right['A'])]
```



### key名不同时
如果连接的两个表在键名不同的列上进行 join 操作，需要使用 left_on 和 right_on 参数显式指定左右两侧连接的列名。

例如：

```Python
import pandas as pd

df1 = pd.DataFrame({'k': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

outer_join = pd.merge(df1, df2, how='outer', left_on='k', right_on='key')
print(outer_join)
```

以上代码执行后，输出结果为：
outer
```python
    k  value1  key  value2
0   A     1.0  NaN     NaN
1   B     2.0    B     5.0
2   C     3.0  NaN     NaN
3   D     4.0    D     6.0
4 NaN     NaN    E     7.0
5 NaN     NaN    F     8.0
```

inner
```python
   k  value1 key  value2
0  B       2   B       5
1  D       4   D       6
```

left
```python
   k  value1  key  value2
0  A       1  NaN     NaN
1  B       2    B     5.0
2  C       3  NaN     NaN
3  D       4    D     6.0
```

right
```python
     k  value1 key  value2
0    B     2.0   B       5
1    D     4.0   D       6
2  NaN     NaN   E       7
3  NaN     NaN   F       8
```

df1 = pd.merge(df1, df_freeze, how='left', left_on='交易卡号', right_on='账卡号')
会出现左侧重复行，匹配右侧数据存在对应问题
pandas.errors.MergeError: Not allowed to merge between different levels. (2 levels on the left, 1 on the right)