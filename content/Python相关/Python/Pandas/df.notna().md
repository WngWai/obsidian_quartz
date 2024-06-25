是一个方法，用于检查数据框（DataFrame）中的**每个元素是否为非缺失值**，并返回一个**布尔值的数据框**，其中对应位置为 True 表示非缺失值，False 表示缺失值。

```python
import pandas as pd
import numpy as np

# 创建一个包含缺失值的数据框
data = {'Name': ['John', 'Alice', np.nan, 'Bob'],
        'Age': [25, np.nan, 35, 40],
        'City': ['New York', 'Paris', 'London', np.nan]}
df = pd.DataFrame(data)

print("原始数据框：")
print(df)

# 检查非缺失值
notna_df = df.notna()

print("\n非缺失值的数据框：")
print(notna_df)
```

在上面的示例中，我们首先创建了一个包含缺失值的数据框 `df`，其中 NaN 表示缺失值。然后，我们调用 `df.notna()` 方法，将返回一个布尔值的数据框 `notna_df`，其中对应位置为 True 表示非缺失值，False 表示缺失值。最后，我们打印了原始数据框和非缺失值的数据框。

```python
原始数据框：
   Name   Age      City
0  John  25.0  New York
1  Alice   NaN     Paris
2   NaN  35.0    London
3   Bob  40.0       NaN

非缺失值的数据框：
    Name    Age   City
0   True   True   True
1   True  False   True
2  False   True   True
3   True   True  False
```

可以看到，在非缺失值的数据框中，对应位置为 True 表示该位置的元素为非缺失值，False 表示该位置的元素为缺失值。例如，在原始数据框中，Name 列的第一行和第四行都有值，所以在非缺失值的数据框中对应位置为 True。而在 Age 列的第二行有缺失值，所以在非缺失值的数据框中对应位置为 False。