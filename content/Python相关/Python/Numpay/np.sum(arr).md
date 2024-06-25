`np.sum()`是NumPy库中的函数，它可以用来计算NumPy数组中**所有元素的总和**，也可以用来计算多维数组的某个轴上的元素总和。它的参数可以是一个NumPy数组，也可以是一个Python列表、元组等可迭代对象。下面是一个例子：

```python
import numpy as np

# 创建一个NumPy数组
arr = np.array([1, 2, 3, 4, 5])

# 计算数组中所有元素的总和
total = np.sum(arr)

print(total)
```

输出结果为：

```
15
```
