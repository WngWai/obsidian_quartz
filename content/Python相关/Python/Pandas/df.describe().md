
![[1685842782757.jpg|400]]

`describe()` 是 Pandas 中的一个函数，用于计算数据集的统计信息，包括计数、均值、标准差、最小值、25% 分位数、中位数、75% 分位数和最大值。该函数默认只计算数值型数据的统计信息，可以通过设置参数 `include` 和 `exclude` 来指定计算哪些列的统计信息。

函数语法如下：

```python
DataFrame.describe(percentiles=None, include=None, exclude=None)
```

参数说明：

- `percentiles`：指定要计算的分位数，默认为 `[.25, .5, .75]`，即计算 25%、50% 和 75% 的分位数。
- `include`：指定要**计算的列的数据类型**，可以是 `None`、`'all'`、数据类型或数据类型列表。默认为 `None`，即**只计算数值型列**的统计信息。如果设置为 `'all'`，则计算**所有列的统计信息**；如果设置为数据类型或数据类型列表，则只计算指定数据类型的列的统计信息。
df.describe(include=['O'])，'O' 表示 object 数据类型，通常用于字符串数据

- `exclude`：指定不计算的列的数据类型，可以是数据类型或数据类型列表。默认为 `None`，即计算所有列的统计信息。如果设置为数据类型或数据类型列表，则不计算指定数据类型的列的统计信息。

例如，假设有一个数据集 `df`，可以通过以下代码计算其数值型列的统计信息：

```python
import pandas as pd

df = pd.read_csv('data.csv')
df.describe()
```

如果要计算所有列的统计信息，可以将 `include` 参数设置为 `'all'`：

```python
df.describe(include='all')
```

如果要计算指定数据类型的列的统计信息，可以将 `include` 参数设置为数据类型或数据类型列表：

```python
df.describe(include=['object', 'int64'])
```

如果要排除指定数据类型的列的统计信息，可以将 `exclude` 参数设置为数据类型或数据类型列表：

```python
df.describe(exclude=['float64'])
```