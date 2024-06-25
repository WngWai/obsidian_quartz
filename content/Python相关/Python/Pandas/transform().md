`transform()` 函数是 pandas 库中的一个非常强大且灵活的函数，它主要用于**执行组级别的操作**，并将结果**广播到原始 DataFrame 的相应行上**。通常，`transform()` 用于对每个组应用函数，然后将每个组的结果广播回原始 DataFrame。以下是有关 `transform()` 函数的详细信息：

**所属包：** pandas

**定义：**
```python
DataFrame.transform(func, axis=0, *args, **kwargs)
```

**参数介绍：**
- `func`：函数或函数列表。用于在每个组上执行的函数。可以是标量函数、字符串函数（表示已定义的方法名），或者是函数列表。

- `axis`：指定应用函数的轴。默认为0，表示在每一列上应用函数；1 表示在每一行上应用函数。

- `*args` 和 `**kwargs`：用于传递给 `func` 的其他参数。

**返回值：**
- 与输入 DataFrame 具有相同形状和索引的对象，其中包含应用了函数的结果。

**示例：**
```python
import pandas as pd

# 创建一个DataFrame
data = {'Group': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 15, 20, 25, 30]}
df = pd.DataFrame(data)

# 定义一个函数，计算每个组的平均值
def group_mean(x):
    return x.mean()

# 使用 transform 应用 group_mean 函数
df['Group_Mean'] = df.groupby('Group')['Value'].transform(group_mean)

print(df)
```

**输出：**
```
  Group  Value  Group_Mean
0     A     10          20
1     B     15          20
2     A     20          20
3     B     25          20
4     A     30          20
```

在上述示例中，`groupby('Group')['Value'].transform(group_mean)` 对 'Value' 列进行分组，并计算每个组的平均值，然后将这些平均值广播到原始 DataFrame 中的相应行。