用于按照**指定的方式将数组进行拆分**。具体而言，它将一个数组拆分为多个子数组，并返回一个由这些子数组组成的列表。

```python
np.split(ary, indices_or_sections, axis=0)
```

- ary: 要拆分的数组。
- indices_or_sections: 划分的位置。可以是整数、列表或元组。如果是**整数n**，则将数组拆分为**n个相等大小**的子数组。如果是**列表或元组**，则表示要按照**指定的位置**进行拆分。
- axis: 拆分的轴，**默认为0**，即按照第一个维度（行）进行拆分。在多维数组上指定轴进行拆分

### 将数组拆分为相等大小的子数组
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

result = np.split(arr, 3)

print(result)

输出结果：
[array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
```

### 按照指定的位置进行拆分
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

result = np.split(arr, [2, 5])

print(result)

输出结果：
[array([1, 2]), array([3, 4, 5]), array([6, 7, 8, 9])]
```


### 二维数据拆分
```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

result1 = np.split(arr, 3, axis=0)  # 按行进行均匀拆分
result2 = np.split(arr, 3, axis=1)  # 按列进行均匀拆分

print(result1)
print(result2)

输出结果： 
[array([[1, 2, 3]]), array([[4, 5, 6]]), array([[7, 8, 9]])] 

[array([[1], [4], [7]]), array([[2], [5], [8]]), array([[3], [6], [9]])]
```