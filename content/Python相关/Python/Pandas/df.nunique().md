### `nunique()` 函数概述：
返回唯一值的数量，int型数据。
跟df.unique()反馈唯一元素值相区分！

**所属包：** Pandas（需要导入 Pandas 包）

**功能：** `nunique()` 函数用于计算 DataFrame 中每列中唯一值的数量。

**定义：**
```python
DataFrame.nunique(axis=0, dropna=True)
```

### 参数介绍：

- **`axis`：** 指定计算唯一值数量的轴，`0` 表示在每列上计算，`1` 表示在每行上计算。默认为 `0`。

- **`dropna`：** 是否在计算唯一值数量时排除缺失值。默认为 `True`，表示排除。

### 示例：

```python
import pandas as pd

# 创建一个示例数据框
data = {'A': [1, 2, 1, 3, 2],
        'B': [4, 5, 5, 6, 4],
        'C': [7, 8, 9, 9, 7]}

df = pd.DataFrame(data)

# 使用 nunique() 计算每列中的唯一值数量
unique_counts = df.nunique()

# 输出结果
print(unique_counts)
```

### 输出：

```
A    3
B    3
C    3
dtype: int64
```

在这个例子中，`nunique()` 函数被用于计算数据框中每列的唯一值数量。结果是一个 Pandas Series，其中索引是数据框的列名，值是相应列中的唯一值数量。

### 注意事项：

- `nunique()` 函数返回的结果是一个包含唯一值数量的 Pandas Series。

- 可以通过设置 `axis` 参数来在行或列上计算唯一值数量。

- 默认情况下，`dropna` 参数为 `True`，表示在计算唯一值数量时排除缺失值。如果设置为 `False`，则包括缺失值在内。