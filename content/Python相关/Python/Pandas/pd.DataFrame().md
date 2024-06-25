是 Pandas 库中创建**数据框**的主要函数之一。
### 列表或数组
用列表表示**矩阵**内容。

```python
import pandas as pd

# 从列表创建数据框
data1 = ['Alice', 25] # 代表一列元素！
df1 = pd.DataFrame(data1, columns=['Name'])
print(df1)

# 输出
  Name 
0 Alice
1  25

# 从二维数组创建数据框
data1 = [['Alice', 25]] # 代表一行！
df1 = pd.DataFrame(data1, columns=['Name', 'Age'])
print(df1)

# 输出
      Name  Age
0     Alice   25


# 从二维数组创建数据框，数组和列表很相似，主要看开始的定义
data3 = [[25, 50, 75], 
		 [30, 60, 90], 
		 [35, 70, 105]]
df3 = pd.DataFrame(data3, index=['Alice', 'Bob', 'Charlie'], columns=['Math', 'English', 'Science'])
print(df3) 

# 输出：
         Math  English  Science
Alice      25       50       75
Bob        30       60       90
Charlie    35       70      105
```
### 字典
函数里面套了字典！
```python
data = pd.DataFrame({'month': [1, 4, 7, 10],  
                     'year': [2012, 2014, 2013, 2014],  
                     'sale': [55, 40, 84, 31]})
```

行索引自动添加
```python
   month  year  sale
0      1  2012    55
1      4  2014    40
2      7  2013    84
3     10  2014    31
```

### series
这个series还是字典的形式，列名为key值，行名用pd.Series()函数定义。更多的用上面的数据或则字典定义。

```python

# 从 Series 创建数据框
data4 = {'Math': pd.Series([25, 30, 35], 
		 index=['Alice', 'Bob', 'Charlie']),
         'English': pd.Series([50, 60, 70],
         index=['Alice', 'Bob', 'Charlie']),
       'Science': pd.Series([75, 90, 105], 
         index=['Alice', 'Bob', 'Charlie'])}
df4 = pd.DataFrame(data4)
print(df4)
```

```python
         Math  English  Science
Alice      25       50       75
Bob        30       60       90
Charlie    35       70      105
```


### []和[[]]的区别
在创建DataFrame时，如果**data是一个列表，Pandas会将这个列表视为一列数据**。

如果你想创建一个DataFrame，其中'data1' 是一行数据，你可以将它放在另一个列表中：

```python
data1 = [['Alice', 25]]
df1 = pd.DataFrame(data1, columns=['Name', 'Age'])
print(df1)
```

如果你想创建一个DataFrame，其中'data1' 是一列数据，你可以使用字典来创建：

```python
data1 = {'Name': ['Alice'], 'Age': [25]}
df1 = pd.DataFrame(data1)
print(df1)
```

参数作用如下：

- `data`：需要转换成数据框的数据，可以是列表、字典、二维数组、Series 等等。

- `index`：数据框的行索引，可以是列表、数组、Series 等等，默认为 `range(n)`，其中 `n` 为数据框的行数。

- `columns`：数据框的列索引，可以是列表、数组、Series 等等，默认为 `range(n)`，其中 `n` 为数据框的列数。

- `dtype`：数据框中各列的数据类型，可以是字符串、字典等等，默认为 `None`。

- `copy`：是否复制数据，如果为 `True`，则复制原始数据，否则直接引用原始数据，默认为 `False`。

- `...`：其他参数，如 `columns`、`index`、`dtype` 等等。