 是一个属性，用于获取DataFrame索引的**级别值**。它返回一个**元组**，元组中的每个元素是一个包含该级别下所有不重复值的列表。

下面是一个示例：
```python
import pandas as pd

# 创建一个多级索引的DataFrame
data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
index = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'a'), ('y', 'b')], names=['first', 'second'])
df = pd.DataFrame(data=data, index=index)
print(df)

# 获取索引的级别值
print(df.index.levels)
```
输出：
```python
           A  B
first second      
x     a      1  5
      b      2  6
y     a      3  7
      b      4  8


[['x', 'y'], ['a', 'b']]
```

在上面的示例中，我们创建了一个多级索引的DataFrame，并通过参数 `names` 指定了两个索引级别的名称为 `['first', 'second']`。然后使用 `df.index.levels` 获取索引的级别值，返回的是一个元组，其中第一个元素是 `['x', 'y']`，第二个元素是 `['a', 'b']`。这表示第一级索引的所有不重复值为 `['x', 'y']`，第二级索引的所有不重复值为 `['a', 'b']`。

总结来说，`df.index.levels` 是一个属性，用于获取DataFrame索引的级别值。它返回一个元组，每个元素是一个包含该级别下所有不重复值的列表。