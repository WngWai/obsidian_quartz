 是 Pandas DataFrame 对象的除法运算方法，用于对 DataFrame 中的元素进行除法计算。

```python
df.div(other, axis='columns', level=None, fill_value=None)
```


- `other`: 参与除法计算的对象，可以是标量、Series 或 DataFrame。如果是标量，则会将 DataFrame 的每个元素除以该标量；如果是 Series 或 DataFrame，则会进行元素级别的除法计算。
- `axis`: 指定在哪个轴上进行除法计算，默认为 'columns'，即按列进行计算。可以设置为 0 或 'index' 表示按行计算。
- `level`: 当 DataFrame 具有多级索引时，指定要在哪个级别上进行除法计算。
- `fill_value`: 当 DataFrame 中的元素缺失或被除数为 0 时，用于填补缺失值或避免除以 0 的默认值。


### df对df
`df.div()` 方法也可以用于对不同形状的 DataFrame 或 Series 进行元素级别的除法计算。当多个数据结构参与除法运算时，Pandas 会自动进行广播操作。以下是一个示例：

```python
import pandas as pd

data1 = {'A': [10, 20, 30],
         'B': [2, 4, 5]}
df1 = pd.DataFrame(data1)

data2 = {'A': [2, 2, 2],
         'B': [2, 2, 2]}
df2 = pd.DataFrame(data2)

# 对两个 DataFrame 进行除法运算
result = df1.div(df2)
print(result)
```

输出结果为：

```python
     A    B
0  5.0  1.0
1  10.0  2.0
2  15.0  2.5
```

在上面的示例中，我们对两个形状相同的 DataFrame `df1` 和 `df2` 进行除法运算，得到了新的 DataFrame。`df1.div(df2)` 会将 `df1` 的元素除以 `df2` 的对应元素。

### df对标量

```python
import pandas as pd

data = {'A': [10, 20, 30],
        'B': [2, 4, 5]}
df = pd.DataFrame(data)

# 将 DataFrame 的每个元素除以 10
result = df.div(10)
print(result)
```

输出结果为：

```python
     A    B
0  1.0  0.2
1  2.0  0.4
2  3.0  0.5
```

在上面的示例中，我们将 DataFrame `df` 的每个元素除以了 10，得到了新的 DataFrame。由于 10 是一个标量，它会将 `df` 的每个元素进行除法运算。

