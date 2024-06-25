返回**ndarray数组结构**，数组还是包含字符串吗？注意元素间没有逗号！

用于获取数据表中的数值数据，即不包括索引和列标签的数据部分。该属性不需要传入任何参数，直接调用即可。举例如下：

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles']}
        
data = pd.DataFrame(data)

print(data.values)

```

运行结果是：

```python
[['Alice' 25 'New York']  
 ['Bob' 30 'San Francisco']  
 ['Charlie' 35 'Los Angeles']]
```

