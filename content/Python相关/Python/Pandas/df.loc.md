是 Pandas 库中的一种基于**标签**进行**索引**的方法，用于从 DataFrame 中选择指定行和列的数据。
需要注意的是，`df.loc[]` 使用的是基于标签的索引，而不是基于整数位置的索引。

```python
df.loc[row_label, column_label]
```

参数说明：
- `row_label`：行标签，用于选择特定的行或行范围。
- `column_label`：列标签，用于选择特定的列或列范围。

### 示例 1：选择单个元素
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=['a', 'b', 'c'])
print(df.loc['b', 'B'])
```
输出：
```python
5
```
选择行标签为 'b' 和列标签为 'B' 的元素。

### 示例 2：选择行范围
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 'c'])
print(df.loc['b':'c', :])
```
输出：
```python
   A  B  C
b  2  5  8
c  3  6  9
```
选择行标签从 'b' 到 'c'（包含 'c'）的所有行，以及所有列。

### 示例 3：选择列范围
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=['a', 'b', 'c'])
print(df.loc[:, 'A':'B'])
```
输出：
```python
   A  B
a  1  4
b  2  5
c  3  6
```
选择所有行以及列标签从 'A' 到 'B'（包含 'B'）的所有列。

### 示例 4：选择特定行和列
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, index=['a', 'b', 'c'])
print(df.loc[['a', 'c'], ['B', 'C']])
```
输出：
```python
   B  C
a  4  7
c  6  9
```
选择行标签为 'a' 和 'c' 的行，以及列标签为 'B' 和 'C' 的列。

在这些示例中 ，我们展示了如何使用 `df.loc[]` 方法从 DataFrame 中选择指定的行和列。你可以根据需要使用适当的行标签和列标签来选择数据。

### 自己用的例子
```python
print(stock.loc["股票--0", :]) # 横向取片，可以指定行标签名字  
print(stock.loc[stock.index[0], :]) # 实质同上  
print(stock.loc[:, "2023-05-29"], end="\n") # 竖向取片，可以指定列标签名字

women = train_data.loc[train_data.Sex == 'female']["Survived"]
# 先行后列
women = train_data[train_data.Sex == 'female'].loc[:, "Survived"] # 或者这样

```