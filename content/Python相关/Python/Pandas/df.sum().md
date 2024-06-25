`np.sum()`和[[np.sum(arr)]]都是用来计算数组或DataFrame中所有元素的总和，但是它们的参数和用法有所不同。

`df.sum()`是Pandas库中DataFrame对象的方法，它可以用来计算DataFrame中所有数值型列的总和，也可以用来计算某些列的总和。它的参数可以是一个**轴标签**（例如'index'或'columns'），也可以是一个**列名**或一个**列名列表**。下面是一个例子：

```python
import pandas as pd

# 创建一个DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# 计算所有数值型列的总和
total1 = df.sum()

# 计算某些列的总和
total2 = df[['A', 'C']].sum()

print(total1)
print(total2)
```

输出结果为：

```
A     6
B    15
C    24
dtype: int64

A     4
C    12
dtype: int64
```

在上面的例子中，我们使用了`df.sum()`方法来计算DataFrame中所有数值型列的总和以及某些列的总和，并将结果保存在了`total1`和`total2`两个变量中。注意到`df.sum()`方法的返回值是一个**Series对象**，其中每个元素对应一个列的总和。

