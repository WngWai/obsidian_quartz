是 Pandas 库中 Series 对象的一个属性，用于获取 Series 对象中的数据部分（值部分）。

在 Pandas 中，Series 是一维标签化的数组，由两部分组成：索引（行标签）和数据（值）。`sr.values` 属性返回的是 Series 对象的数据（值）部分，以 Numpy 数组的形式呈现。

以下是一个示例，展示如何使用 `sr.values` 来获取 Series 对象的值：

```python
import pandas as pd

# 创建一个 Series 对象
data = [1, 2, 3]
index = ['a', 'b', 'c']
sr = pd.Series(data, index=index)

# 获取 Series 对象的值
print(sr.values)
```

输出结果为：
```python
[1 2 3]
```

上述示例中，首先创建了一个包含三个元素的 Series 对象 `sr`，其中的数据为 `[1, 2, 3]`，索引为 `['a', 'b', 'c']`。然后，使用 `sr.values` 来获取 `sr` 对象的值，返回的结果是一个 Numpy 数组，其中包含了实际的数据值 `[1, 2, 3]`。
