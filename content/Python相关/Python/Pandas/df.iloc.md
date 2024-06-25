是 Pandas 库中的一种基于**整数位置**进行索引的方法，用于从 DataFrame 中选择指定行和列的数据。
以下是 `df.iloc[]` 的语法：
```python
df.iloc[row_index, column_index]
```
参数说明：
- `row_index`：行索引，用于选择特定的行或行范围。
- `column_index`：列索引，用于选择特定的列或列范围。

需要注意的是，`df.iloc[]` 使用的是基于 0 的整数位置索引，而不是基于标签的索引。

下面是一些 `df.iloc[]` 的示例：

### 示例 1：选择单个元素
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print(df.iloc[0, 1])
```
输出：
```python
4
```
选择第一行（索引为0）和第二列（索引为1）的元素。

### 示例 2：选择行范围
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print(df.iloc[1:3, :])
```
输出：
```python
   A  B  C
1  2  5  8
2  3  6  9
```
选择索引为 1 到 2 的行（不包含索引 3）的所有列。

### 示例 3：选择列范围
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print(df.iloc[:, 0:2])
```
输出：
```python
   A  B
0  1  4
1  2  5
2  3  6
```
选择所有行的索引为 0 到 1 的列（不包含索引 2）。

### 示例 4：选择特定行和列
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print(df.iloc[[0, 2], [1, 2]])
```
输出：
```python
   B  C
0  4  7
2  6  9
```
选择索引为 0 和 2 的行和索引为 1 和 2 的列。

在这些示例中，我们展示了如何使用 `df.iloc[]` 方法从 DataFrame 中选择指定的行和列。你可以根据需要使用适当的行索引和列索引来选择数据。

自己使用的例子
```python
print(stock.iloc[0, :]) # 横向取片，指定行标签位置。  
print(stock.iloc[:, 0]) # 竖向取片，指定列标签位置。  
print(stock.iloc[[1, 2], 0], end="\n") # 指定第一列，第二、三行
```