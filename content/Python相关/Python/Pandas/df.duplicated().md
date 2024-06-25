在 Python 中，`df.duplicated()` 函数是 Pandas 库中的方法，用于**标识 DataFrame 中的重复行**。

True标记重复行。输出**布尔型 Series**

**所属包：** Pandas

**定义：**
```python
DataFrame.duplicated(subset=None, keep='first')
```

**参数介绍：**
- `subset`：可选，指定要**考虑重复的列或列的组合**。默认为 `None`，表示考虑**所有列**。

- `keep`：可选，指定保留哪个重复项。可以是 `'first'`（保留第一个出现的）、`'last'`（保留最后一个出现的）。

**功能：**
返回一个布尔型 Series，表示每一行是否为重复行。

**举例：**
```python
import pandas as pd

# 创建一个包含重复行的 DataFrame
data = {'A': [1, 2, 2, 3, 4, 4],
        'B': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar']}
df = pd.DataFrame(data)

# 使用 duplicated() 标识重复行
duplicates_mask = df.duplicated()

# 打印结果
print(duplicates_mask)
```

**输出：**
```
0    False
1    False
2     True
3    False
4    False
5     True
dtype: bool
```

在这个例子中，`duplicated()` 函数被用于标识 DataFrame `df` 中的重复行。输出是一个布尔型 Series，其中每个元素表示相应行是否为重复行。在输出中，第三行和第六行被标记为重复行。