用于获取数据表的列名，即列标签。该属性不需要传入任何参数，直接调用即可。举例如下：

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b'])
print(df.columns)
```

运行结果是：

```python
Index(['A', 'B'], dtype='object')
```

