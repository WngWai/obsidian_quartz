在Python中，`df.unique()`函数通常用于获取DataFrame中某一列的唯一值。这个函数是pandas库中DataFrame对象的方法。

以下是有关`df.unique()`函数的一些详细信息：

**所属包：** pandas

**定义：**
```python
DataFrame.unique(subset=None, keep='first')
```

**参数介绍：**
- `subset`（可选）：指定要获取唯一值的列或列的子集。默认为None，表示使用整个DataFrame的所有列。
- `keep`（可选）：指定保留哪个重复项的唯一值。默认为'first'，表示保留第一个出现的值。其他可能的值包括'last'（保留最后一个出现的值）和`False`（不保留重复项）。

**举例：**
```python
import pandas as pd

# 创建一个DataFrame
data = {'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Age': [25, 30, 25, 35, 30]}
df = pd.DataFrame(data)

# 获取 'Name' 列的唯一值
unique_names = df['Name'].unique()

print(unique_names)
```

**输出：**
```
['Alice' 'Bob' 'Charlie']
```

在上面的例子中，`df['Name'].unique()`返回了 'Name' 列中的唯一值数组。

需要注意的是，返回的结果是一个NumPy数组（numpy.ndarray）。

希望这能帮助你理解`df.unique()`函数的基本用法和功能。