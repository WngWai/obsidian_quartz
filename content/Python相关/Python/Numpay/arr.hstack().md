是NumPy库中的一个函数，用于**将两个或多个数组水平**地堆叠在一起，生成一个新的数组。


参数：
- tup: 一个元组，包含要堆叠在一起的数组。


```python
import numpy as np

# 创建两个数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 使用np.hstack()函数进行水平堆叠
result = np.hstack((arr1, arr2))

print(result)

输出结果：
[1 2 3 4 5 6]
```

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

result = np.hstack((arr1, arr2, arr3))

print(result)


输出结果：
[1 2 3 4 5 6 7 8 9]
```

