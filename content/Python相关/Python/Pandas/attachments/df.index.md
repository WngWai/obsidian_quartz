返回的是**行标签**

Index对象是一种类似于数组、但具有一定特殊性质的pandas对象，它可以被看作是一种有序集合，其中每个元素都具有唯一性，并且可以进行集合运算或索引操作
用于获取数据表的索引，即行标签。该属性不需要传入任何参数，直接调用即可。举例如下：

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['a', 'b'])
print(df.index)
```

运行结果是：

```python
Index(['a', 'b'], dtype='object')
```
