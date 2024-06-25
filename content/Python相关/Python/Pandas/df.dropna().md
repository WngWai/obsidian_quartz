`dropna`是Pandas中用于删除DataFrame中缺失值的函数。具体用法如下：

```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
```

参数说明：

- `axis`：指定删除缺失值的行或列。当`axis=0`时，删除包含缺失值的行；当`axis=1`时，删除包含缺失值的列。默认为`axis=0`。
- `how`：指定删除的方式。当`how='any'`时，只要该行或列中存在缺失值，就删除；当`how='all'`时，只有该行或列中所有元素都是缺失值时才删除。**默认**为`how='any'`。
- `thresh`：指定删除行或列的缺失值数量阈值。当缺失值数量小于等于该值时，不会被删除。默认为`thresh=None`，即不考虑缺失值数量。
- `subset`：指定删除行或列的子集。可以是列名或行索引。默认为`subset=None`，即删除整行或整列。
- `inplace`：是否在原DataFrame上进行修改。默认为`inplace=False`，即不在原DataFrame上进行修改，而是返回一个新的DataFrame。

例如，假设有一个包含缺失值的DataFrame：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8], 'C': [9, 10, 11, np.nan]})
```

可以使用`dropna`函数删除包含缺失值的行：

```python
df.dropna(axis=0)
```

输出结果为：

```python
     A    B     C
0  1.0  5.0   9.0
3  4.0  8.0   NaN
```

可以使用`dropna`函数删除包含缺失值的列：

```python
df.dropna(axis=1)
```

输出结果为：

```python
     A
0  1.0
1  2.0
2  NaN
3  4.0
```

可以使用`thresh`参数指定删除行或列的缺失值数量阈值：

```python
df.dropna(axis=0, thresh=2)
```

输出结果为：

```python
     A    B     C
0  1.0  5.0   9.0
1  2.0  NaN  10.0
3  4.0  8.0   NaN
```

可以使用`subset`参数指定删除行或列的子集：

```python
df.dropna(axis=0, subset=['B'])
```

输出结果为：

```python
     A    B     C
0  1.0  5.0   9.0
2  NaN  7.0  11.0
3  4.0  8.0   NaN
```

可以使用`inplace`参数在原DataFrame上进行修改：

```python
df.dropna(axis=0, inplace=True)
```

此时，原DataFrame中的缺失值已经被删除。