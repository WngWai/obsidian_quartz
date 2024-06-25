是NumPy库中用于改变数组数据类型的方法。

用法：
```python
ndarray.astype(dtype, casting='unsafe', copy=True)
```

参数：
- `dtype`：新的数据类型。可以是NumPy的数据类型对象，或者是对应的字符串表示。
- `casting`（可选）：指定数据类型转换的方式，有 safe、same_kind、unsafe 和 no 四种选项。
- `copy`（可选）：是否复制数组。默认为**True**，表示**复制**数组。

示例：
```python
import numpy as np

# 创建一个整数类型的数组
arr1 = np.array([1, 2, 3, 4, 5])
print("原数组：", arr1)
print("数据类型：", arr1.dtype)

# 转换为浮点类型
arr2 = arr1.astype(float)
print("转换后的数组：", arr2)
print("数据类型：", arr2.dtype)

# 创建一个浮点类型的数组
arr3 = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
print("原数组：", arr3)
print("数据类型：", arr3.dtype)

# 转换为整数类型
arr4 = arr3.astype(int)
print("转换后的数组：", arr4)
print("数据类型：", arr4.dtype)
```

输出结果：
```python
原数组： [1 2 3 4 5]
数据类型： int64
转换后的数组： [1. 2. 3. 4. 5.]
数据类型： float64
原数组： [1.1 2.2 3.3 4.4 5.5]
数据类型： float64
转换后的数组： [1 2 3 4 5]
数据类型： int64
```

在上述示例中，我们使用了 `nd.astype()` 方法改变数组的数据类型。

首先，我们创建了一个整数类型的数组 `arr1`，然后使用 `astype()` 将其转换为浮点类型的数组 `arr2`。转换后，我们可以看到数组中的元素被转换为浮点数，并且数据类型由 `int64` 变为 `float64`。

接下来，我们创建了一个浮点类型的数组 `arr3`，并使用 `astype()` 将其转换为整数类型的数组 `arr4`。转换后，我们可以看到数组中的元素被转换为整数，并且数据类型由 `float64` 变为 `int64`。

通过改变数组的数据类型，我们可以调整数组元素的表示范围和精度，以满足特定的计算或存储需求。

需要注意的是，`astype()` 方法返回一个新的数组，而不修改原始数组。如果使用 `copy=False` 参数，那么返回的数组是对原始数组的引用。另外，进行数据类型转换时，需要注意合理选择转换方式，以避免数据丢失或不合理的转换。