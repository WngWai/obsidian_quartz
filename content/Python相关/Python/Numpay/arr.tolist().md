---


---
在 Python 中，`tolist()` 是 NumPy 数组对象的方法，用于**将数组转换为 Python 列表**。

以下是 `tolist()` 方法的基本信息：

**所属包：** NumPy

**定义：**
```python
numpy_array.tolist()
```

**参数介绍：**
该方法没有额外的参数。

**功能：**
将 NumPy 数组转换为 Python 列表。

**举例：**
```python
import numpy as np

# 创建一个 NumPy 数组
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])

# 使用 tolist() 方法将数组转换为列表
python_list = numpy_array.tolist()

# 打印转换后的列表
print(python_list)
```

**输出：**
```
[[1, 2, 3], [4, 5, 6]]
```

在上述示例中，`numpy_array.tolist()` 将 NumPy 数组转换为了 Python 列表。这在需要将 NumPy 数组与其他 Python 数据结构进行交互时非常有用，因为列表是 Python 中最常见的数据结构之一。