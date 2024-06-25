在PyTorch中，`torch.matmul()`函数用于执行**矩阵相乘**操作。它可以用于两个**张量之间的矩阵乘法**，以及**张量与矩阵**之间的乘法。
```python
torch.matmul(input, other, out=None)
```
**参数**：
- `input`：输入**张量**，可以是**二维或多维张量**。
- `other`：另一个输入**张量**，可以是**二维或多维张量**。
- `out`（可选）：输出张量，用于指定结果的存储位置。

**示例**：
```python
import torch

# 两个二维张量的矩阵相乘
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(tensor1, tensor2)
print("两个二维张量的矩阵相乘结果：")
print(result)

# 张量与矩阵的乘法
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = torch.matmul(tensor, matrix)
print("\n张量与矩阵的乘法结果：")
print(result)
```

**输出**：
```
两个二维张量的矩阵相乘结果：
tensor([[19, 22],
        [43, 50]])

张量与矩阵的乘法结果：
tensor([[ 9, 12, 15],
        [19, 26, 33],
        [29, 40, 51]])
```

在上述示例中，我们演示了使用`torch.matmul()`函数执行矩阵相乘操作。

首先，我们创建了两个二维张量 `tensor1` 和 `tensor2`，分别表示两个矩阵。然后，我们调用 `torch.matmul(tensor1, tensor2)` 执行矩阵相乘操作，并将结果存储在 `result` 中。打印输出结果可以看到，两个二维张量的矩阵相乘结果被计算出来。

接下来，我们创建了一个二维张量 `tensor` 和一个二维矩阵 `matrix`。然后，我们调用 `torch.matmul(tensor, matrix)` 执行张量与矩阵之间的乘法，并将结果存储在 `result` 中。打印输出结果可以看到，张量与矩阵的乘法结果被计算出来。

`torch.matmul()`函数在深度学习中的很多任务中都非常常用，例如神经网络中的线性层、卷积神经网络中的卷积操作等。通过调用`torch.matmul()`函数，可以方便地进行矩阵相乘和张量与矩阵之间的乘法运算。



### (1000,2)\*(2)
视为(1000,2)\*(2,1)
对于 `torch.matmul()`，如果右侧是一维张量，它会被视为列向量（形状为 (n, 1)）。因此，对于形状为 (1000, 2) 的左侧矩阵和形状为 (2) 的右侧向量，`torch.matmul()` 会自动进行广播，将右侧向量视为列向量。结果将是一个形状为 (1000, 1) 的张量。

```python
import torch

# 左侧矩阵 A 形状为 (1000, 2)
A = torch.randn(1000, 2)

# 右侧向量 B 形状为 (2)
B = torch.randn(2)

# 使用 torch.matmul() 进行矩阵-向量乘法
result = torch.matmul(A, B)

print(result.shape)  # 输出形状 (1000,)
```

在这个例子中，`torch.matmul(A, B)` 的结果将是一个形状为 (1000,) 的张量。