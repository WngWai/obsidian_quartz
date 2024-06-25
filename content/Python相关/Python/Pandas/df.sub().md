是 Pandas 库中的一个函数，用于在 pandas 的 DataFrame 或 Series 上执行元素级别的减法操作。它用于将两个数据结构进行逐元素相减，并返回结果。

```python
df.sub(other, axis='columns', level=None, fill_value=None)
```

**参数说明：**

- `other`：另一个 **DataFrame**、**Series**、**标量**（如整数或浮点数）或**可广播**（broadcast）的对象。
- `axis`：指定减法的轴方向。默认为 `'columns'`，即按列进行减法操作。
- `level`：在 MultiIndex 中指定要匹配的级别。
- `fill_value`：用于填充缺失值的值。默认为 `None`。


### df-df

```python
import pandas as pd

data = {'A': [1, 2, 3],
        'B': [4, 5, 6]}
df = pd.DataFrame(data)
```

```python
   A  B
0  1  4
1  2  5
2  3  6
```

现在，我们将 DataFrame `df` 减去另一个 DataFrame：

```python
data2 = {'A': [10, 20, 30],
         'B': [40, 50, 60]}
df2 = pd.DataFrame(data2)

result = df.sub(df2)

print(result)
```

输出结果为：

```python
   A  B
0 -9 -36
1 -18 -45
2 -27 -54
```

在上述示例中，我们创建了两个具有相同列名的 DataFrame：`df` 和 `df2`。然后，我们使用 `df.sub(df2)` 对它们进行减法操作。结果 `result` 是一个新的 DataFrame，其中元素是将对应位置的元素相减得到的。


### 缺失值的处理
如果两个 DataFrame 相对应位置有缺失值，可以使用 `fill_value` 参数填充缺失值。例如：

```python
result = df.sub(df2, fill_value=0)

print(result)
```

输出结果为：

```python
   A   B
0 -9 -36
1 -18 -45
2 -27 -54
```

在这个示例中，我们将 `fill_value` 参数设置为 0，当相减的位置有缺失值时，缺失值将被填充为 0。
