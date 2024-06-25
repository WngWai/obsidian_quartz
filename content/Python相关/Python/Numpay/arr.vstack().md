用于将**两个或多个数组垂直**地堆叠在一起。具体而言，它将多个数组按垂直方向进行连接，生成一个新的数组。

参数：
- tup: 一个元组，包含要堆叠在一起的数组。

```python
import numpy as np

# 创建两个数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 使用np.vstack()函数进行垂直堆叠
result = np.vstack((arr1, arr2))

print(result)


输出结果：
[[1 2 3]
 [4 5 6]]
```

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

result = np.vstack((arr1, arr2, arr3))

print(result)

输出结果：
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

