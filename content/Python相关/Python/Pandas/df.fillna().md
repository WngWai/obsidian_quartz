`fillna`是Pandas中用于填充DataFrame中缺失值的函数。具体用法如下：
```python
DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)
```
- `value`：指定填充缺失值的值。可以是标量、字典、Series或DataFrame。默认为`value=None`，即不填充缺失值。
- `method`：指定填充缺失值的方法。可以是`ffill`或`pad`，表示使用**前向填充**；也可以是`bfill`或`backfill`，表示使用**后向填充**。默认为`method=None`，即不使用填充方法。
- `axis`：指定填充缺失值的轴。当`axis=0`时，按列填充；当`axis=1`时，按行填充。默认为`axis=None`，即按列填充。
- `inplace`：是否在原DataFrame上进行修改。默认为`inplace=False`，即不在原DataFrame上进行修改，而是返回一个新的DataFrame。
- `limit`：指定填充缺失值的连续数量限制。当`method`参数不为None时，该参数才有意义。默认为`limit=None`，即不限制连续数量。
- `downcast`：指定填充缺失值时的数据类型。可以是`infer`、`integer`、`signed`、`unsigned`、`float`或`complex`。默认为`downcast=None`，即不进行类型转换。

例如，假设有一个包含缺失值的DataFrame：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8], 'C': [9, 10, 11, np.nan]})
```

可以使用`fillna`函数填充缺失值：

```python
df.fillna(value=0)
```

输出结果为：

```python
     A    B     C
0  1.0  5.0   9.0
1  2.0  0.0  10.0
2  0.0  7.0  11.0
3  4.0  8.0   0.0
```

可以使用`method`参数进行前向或后向填充：

```python
df.fillna(method='ffill')
```

输出结果为：

```python
     A    B     C
0  1.0  5.0   9.0
1  2.0  5.0  10.0
2  2.0  7.0  11.0
3  4.0  8.0  11.0
```

可以使用`axis`参数按行填充：

```python
df.fillna(method='bfill', axis=1)
```

输出结果为：

```python
     A    B     C
0  1.0  5.0   9.0
1  2.0  7.0  10.0
2  7.0  7.0  11.0
3  4.0  8.0   NaN
```

可以使用`inplace`参数在原DataFrame上进行修改：

```python
df.fillna(value=0, inplace=True)
```

此时，原DataFrame中的缺失值已经被填充。