在PyTorch中，`torch.from_numpy()`函数用于将NumPy数组转换为对应的PyTorch张量（Tensor）。
**函数定义**：
```python
torch.from_numpy(ndarray)
```
**参数**：
- `ndarray`（必需）：一个NumPy数组，要转换为PyTorch张量。
**返回值**：
`torch.from_numpy()`函数返回一个PyTorch张量，其数据类型和形状与输入的NumPy数组相匹配。
**注意**：
`torch.from_numpy()`函数不会创建新的数据副本。它会共享内存，即NumPy数组和返回的PyTorch张量之间的数据是共享的。因此，对于返回的张量的修改也会反映在原始NumPy数组上。
**示例**：
以下是使用`torch.from_numpy()`函数的示例：
```python
import numpy as np
import torch

# 创建一个NumPy数组
arr = np.array([1, 2, 3, 4, 5])

# 将NumPy数组转换为PyTorch张量
tensor = torch.from_numpy(arr)

print(tensor)  # 输出: tensor([1, 2, 3, 4, 5])
print(tensor.dtype)  # 输出: torch.int64
print(tensor.shape)  # 输出: torch.Size([5])

# 修改PyTorch张量
tensor[0] = 10

# 查看原始NumPy数组
print(arr)  # 输出: [10  2  3  4  5]
```

在上述示例中，我们首先创建一个NumPy数组 `arr`。

然后，我们使用`torch.from_numpy()`函数将`arr`转换为一个PyTorch张量 `tensor`。

我们可以打印输出张量的值、数据类型和形状。

注意，我们对返回的张量进行修改，将索引为0的元素改为10。

最后，我们打印原始的NumPy数组 `arr`，可以看到对张量的修改也反映在原始NumPy数组上。

`torch.from_numpy()`函数在将NumPy数据与PyTorch混合使用时非常有用，它提供了在两种数据结构之间无缝转换的能力。