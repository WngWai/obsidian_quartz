https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append

As of pandas 2.0, append (previously deprecated) was removed.

```python
df = pd.DataFrame()  
for i in range(df_116.shape[0]):  
    # df.loc[len(df)] = match(df_116.iloc[i, :])  
    # 调用match函数  
    result = match(df_116.iloc[i, :])  
    # 检查结果是否为空DataFrame  
    if not result.empty:  
        # 如果不为空，将其追加到df_matches中  
        df = df._append(result)
```

match函数的返回值可能是一个空的DataFrame，也可能是一个包含多行的DataFrame。在Python中，布尔值（如True和False）会被转换为数值（1和0），这意味着**当match函数返回一个空的DataFrame时，它实际上返回了一个全由False组成的Series**（因为空的DataFrame的所有元素都是False），然后您尝试将这个Series作为一行数据追加到DataFrame中，从而导致valueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().


在Python的Pandas库中，`DataFrame`对象有一个名为`append`的方法，用于将一个或多个`DataFrame`对象添加到现有`DataFrame`的末尾。这可以用于合并数据。
### 定义和参数介绍：
`DataFrame.append` 方法的基本定义如下：
```python
DataFrame.append(other, ignore_index=False, verify_integrity=True)
```
- **other**：必需参数，要追加到当前`DataFrame`的另一个`DataFrame`对象。
- **ignore_index**：可选参数，布尔值，如果为`True`，将重置索引。默认值为`False`。
- **verify_integrity**：可选参数，布尔值，如果为`True`，将在添加操作后检查数据的完整性。默认值为`True`。
### 应用举例：
以下是一些使用 `DataFrame.append` 方法的基本示例：
**追加一个DataFrame：**
```python
import pandas as pd
# 创建两个DataFrame
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
# 将df2追加到df1
df_combined = df1.append(df2)
print(df_combined)
```
输出：
```
   A   B
0  1   4
1  2   5
2  3   6
0  7  10
1  8  11
2  9  12
```
在这个例子中，`df1`和`df2`被追加在一起，产生了`df_combined`。注意，索引没有重置，因为默认的`ignore_index`参数是`False`。
**追加多个DataFrame：**
```python
# 创建第三个DataFrame
df3 = pd.DataFrame({'A': [13, 14, 15], 'B': [16, 17, 18]})
# 将df2和df3追加到df1
df_combined = df1.append([df2, df3])
print(df_combined)
```
输出：
```
   A   B
0  1   4
1  2   5
2  3   6
0  7  10
1  8  11
2  9  12
0 13  16
1 14  17
2 15  18
```
在这个例子中，`df1`、`df2`和`df3`被追加在一起，产生了`df_combined`。索引没有重置，因为默认的`ignore_index`参数是`False`。
**重置索引：**
```python
# 将df2追加到df1，并重置索引
df_combined = df1.append(df2, ignore_index=True)
print(df_combined)
```
输出：
```
   index  A   B
0       0  1   4
1       1  2   5
2       2  3   6
0       0  7  10
1       1  8  11
2       2  9  12
```
在这个例子中，`df1`和`df2`被追加在一起，并且索引被重置为从0开始的整数索引，因为`ignore_index`参数被设置为`True`。
