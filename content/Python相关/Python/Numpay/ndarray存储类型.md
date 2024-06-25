NumPy 的 ndarray 对象可以包含不同类型的元素，不仅限于数值类型。ndarray 中可以存储整数、浮点数、复数、布尔值以及其他 Python 对象。

在创建 ndarray 对象时，可以使用 `dtype` 参数指定所需的数据类型。例如，可以使用 `int`、`float`、`bool` 或 `object` 等来声明数组的数据类型。

以下是一些示例：

```python
import numpy as np

# 存储整数数组
arr1 = np.array([1, 2, 3])

# 存储浮点数数组
arr2 = np.array([1.0, 2.5, 3.7])

# 存储布尔数组
arr3 = np.array([True, False, True])

# 存储复数数组
arr4 = np.array([1 + 1j, 2 + 2j, 3 + 3j])

# 存储字符串数组
arr5 = np.array(['a', 'b', 'c'])

# 存储对象数组
arr6 = np.array([1, 'a', True])

print(arr1.dtype)  # int64
print(arr2.dtype)  # float64
print(arr3.dtype)  # bool
print(arr4.dtype)  # complex128
print(arr5.dtype)  # <U1 (unicode 字符串)
print(arr6.dtype)  # object
```

如上所示，ndarray 可以存储多种类型的数据。不过需要注意的是，由于 ndarray 是固定类型的数组，当在数组中包含不同类型的元素时，其元素将被强制转换为最通用的类型，通常是 `object` 类型。