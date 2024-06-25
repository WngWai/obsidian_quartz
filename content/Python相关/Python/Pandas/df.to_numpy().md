在Pandas中，`df.to_numpy()`函数用于将DataFrame**转换为NumPy数组**。它返回一个表示DataFrame数据的二维NumPy数组。
```python
df.to_numpy(dtype=None, copy=False)
```
- `dtype`（可选）：指定返回的NumPy数组的数据类型。默认为None，表示根据DataFrame中的数据类型自动推断。
- `copy`（可选）：指定是否返回数组的副本。默认为False，表示不进行拷贝。
**示例**：
```python
import pandas as pd

# 创建一个示例DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Paris', 'London']}
df = pd.DataFrame(data)

# 将DataFrame转换为NumPy数组
numpy_array = df.to_numpy()
print("DataFrame转换为NumPy数组：")
print(numpy_array)
```

**输出**：
```
DataFrame转换为NumPy数组：
[['Alice' 25 'New York']
 ['Bob' 30 'Paris']
 ['Charlie' 35 'London']]
```

在上述示例中，我们首先创建了一个示例的DataFrame `df`，其中包含了三个列：`Name`、`Age`和`City`。然后，我们使用`df.to_numpy()`函数将DataFrame转换为NumPy数组。

我们调用 `df.to_numpy()`，并将返回的数组存储在 `numpy_array` 中，并打印输出。可以看到，DataFrame转换为了一个二维的NumPy数组，其中每一行对应DataFrame的一行，每一列对应DataFrame的一个列。

需要注意的是，`df.to_numpy()`函数返回的NumPy数组不包含DataFrame的行索引和列标签，只包含数据值。如果需要保留行索引和列标签，可以使用 `df.values` 属性来获取一个二维的NumPy数组。

`df.to_numpy()`函数在将DataFrame数据转换为NumPy数组时非常有用，可以方便地利用NumPy的功能进行数据分析和处理。然而，需要注意的是，将大型DataFrame转换为NumPy数组可能会占用大量内存，因此在处理大型数据时要谨慎使用。