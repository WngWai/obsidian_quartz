是 Pandas 库中 Series 对象的一个属性，用于获取 Series 对象的**索引（行标签）**。

索引是用于对 Series 对象中的每个元素进行唯一标识和访问的标签。它可以是整数、字符串、日期等等。

以下是一个示例，展示如何使用 `sr.index` 来获取 Series 对象的索引：

```python
import pandas as pd

# 创建一个 Series 对象
data = [1, 2, 3]
index = ['a', 'b', 'c']
sr = pd.Series(data, index=index)

# 获取 Series 对象的索引
print(sr.index)
```

输出结果为：
```
Index(['a', 'b', 'c'], dtype='object')
```

上述示例中，首先创建了一个包含三个元素的 Series 对象 `sr`，其中的数据为 `[1, 2, 3]`，索引为 `['a', 'b', 'c']`。然后，使用 `sr.index` 来获取 `sr` 对象的索引，返回的结果是一个 `Index` 对象，其中包含了索引的值和数据类型。在此示例中，索引值为 `['a', 'b', 'c']`，数据类型为 `object`。

通过使用 `sr.index` 属性，可以方便地获取 Series 对象的索引，以便进一步进行数据访问、处理和分析。