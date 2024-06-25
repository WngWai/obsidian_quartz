 是 NumPy 库中的一个函数，用于从文本文件中加载数据到 NumPy 数组中。**nan**显示为缺失值

```python
np.genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, 
names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True)
```

- `fname`：**字符串或文件对象**，指定要读取的文件名或文件。
- `dtype`：可选参数，指定生成的**数组的数据类型**。
- `comments`：可选参数，指定要忽略的注释字符。
- `delimiter`：可选参数，指定**字段的分隔符**。
- `skip_header`：可选参数，指定要跳过的文件头行数。
- `skip_footer`：可选参数，指定要跳过的文件尾行数。
- `converters`：可选参数，指定用于将特定列转换为特定数据类型的函数字典。
- `missing_values`：可选参数，指定要处理的缺失值。
- `filling_values`：可选参数，指定用于填充缺失值的值。
- `usecols`：可选参数，指定要读取的列。
- `names`：可选参数，指定生成的结构化数组的字段名。
- `excludelist`：可选参数，指定要排除的列名列表。
- 其他参数（例如 `deletechars`、`autostrip` 等）控制读取数据的特定行为。

```python
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1, dtype=float)

print(data)
```

在上述示例中，我们从名为 `data.csv` 的文件中加载数据。使用逗号作为字段的分隔符，并跳过文件的第一行（文件头）。最后，将数据加载到 `data` 数组中，并打印输出。

`np.genfromtxt()` 可以处理包含文本和数值数据的文件，并根据提供的参数自动推断数据类型。函数还提供了很多选项来控制数据的加载和转换过程，以满足不同的需求。