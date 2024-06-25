在pandas中，`df.iterrows()`函数用于迭代DataFrame的每一行。它返回一个包含**行索引和行数据的元组**，可以用于在循环中逐行处理DataFrame的数据。
**函数定义**：
```python
DataFrame.iterrows()
```
**参数**：
`df.iterrows()`函数没有接受任何参数。
**示例**：
```python
import pandas as pd

# 示例：迭代DataFrame的每一行并打印行数据
data = pd.DataFrame({
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
})

# 使用iterrows()迭代每一行并打印行数据
for index, row in data.iterrows():
    print('Row index:', index)
    print('Row data:')
    print(row)
    print('------')
```
这个返回的元组带有列标签属性，可以当作单行的df！知识行标签返回给了参数index！
```python
Row index: 0
Row data:
Name      John
Age         25
City    New York
Name: 0, dtype: object
------
Row index: 1
Row data:
Name     Alice
Age          30
City     London
Name: 1, dtype: object
------
Row index: 2
Row data:
Name      Bob
Age        35
City    Paris
Name: 2, dtype: object
------
```

在示例中，我们创建了一个包含"Name"、"Age"和"City"三列的数据框`data`。我们使用`iterrows()`函数迭代每一行，并将返回的元组解包为`index`和`row`。在循环中，我们打印行索引和行数据，以便逐行处理DataFrame的数据。

请注意，`df.iterrows()`函数返回的行数据是一个Series对象，其中包含每一列的值。您可以使用`row['ColumnName']`来访问特定列的值，例如`row['Name']`、`row['Age']`等。然而，由于`df.iterrows()`函数的迭代性能较低，如果您需要对整个DataFrame进行操作，通常推荐使用矢量化的操作，而不是使用`iterrows()`函数。