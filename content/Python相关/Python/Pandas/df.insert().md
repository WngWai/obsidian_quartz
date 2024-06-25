在 Pandas 中，`df.insert()` 函数用于在 DataFrame 中**插入一列或多列数据**。它的语法如下：

```python
df.insert(loc, column, value, allow_duplicates=False)
```

参数的详细讲解如下：

- `loc`：**，是一个整数值。例如，如果 `loc=0`，表示在 DataFrame 的第一列位置插入数据。
- `column`：要**插入的列的名称**，可以是一个**字符串或整数**。例如，`column='A'` 或 `column=0`。
- `value`：要**插入的数据**，可以是一个标量值、列表、数组或 Series 对象。数据的长度必须与 DataFrame 的行数相同。
- `allow_duplicates`：可选参数，用于指定**是否允许插入重复的列名**。默认为 False，表示不允许插入重复的列名。

下面是一些示例，以展示 `df.insert()` 函数的使用：

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# 在第二列位置插入一列新数据
df.insert(1, 'Gender', ['Female', 'Male', 'Male'])
print(df)
'''
输出：
      Name  Gender  Age
0    Alice  Female   25
1      Bob    Male   30
2  Charlie    Male   35
'''

# 在最后一列位置插入一列新数据
df.insert(len(df.columns), 'Salary', [5000, 6000, 7000])
print(df)
'''
输出：
      Name  Gender  Age  Salary
0    Alice  Female   25    5000
1      Bob    Male   30    6000
2  Charlie    Male   35    7000
'''

# 插入多列数据
df.insert(2, 'City', ['London', 'New York', 'Paris'])
df.insert(3, 'Country', ['UK', 'USA', 'France'])
print(df)
'''
输出：
      Name  Gender      City Country  Age  Salary
0    Alice  Female    London      UK   25    5000
1      Bob    Male  New York     USA   30    6000
2  Charlie    Male     Paris  France   35    7000
'''
```

在以上示例中，我们首先创建了一个 DataFrame `df`，然后使用 `df.insert()` 函数在指定位置插入了新的列数据。在第一个示例中，我们在第二列位置插入了一列名为 'Gender' 的数据。在第二个示例中，我们在最后一列的位置插入了一列名为 'Salary' 的数据。在第三个示例中，我们插入了两列新数据，分别是 'City' 和 'Country'。

通过使用 `df.insert()` 函数，可以方便地在 DataFrame 中插入新的列数据，并指定插入位置和列名。这在数据处理和特征工程中非常有用。