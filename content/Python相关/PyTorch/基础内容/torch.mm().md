`torch.mm()` 是 PyTorch 中用于执行**两个矩阵的乘法**的函数。它接受两个二维矩阵作为输入，并返回它们的矩阵乘积结果。
```python
torch.mm(mat1, mat2, out=None)
```
  - `mat1`：输入矩阵1，形状为 (M, K)。
  - `mat2`：输入矩阵2，形状为 (K, N)。
  - `out`：可选参数，用于指定输出 Tensor。

下面是一个示例：
```python
import torch

mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[5, 6], [7, 8]])

result = torch.mm(mat1, mat2)
print(result)
# 输出: tensor([[19, 22],
#               [43, 50]])
```

在这个例子中，`torch.mm(mat1, mat2)` 对输入的矩阵 `mat1` 和 `mat2` 进行矩阵乘法操作。矩阵 `mat1` 的形状为 (2, 2)，矩阵 `mat2` 的形状为 (2, 2)，因此乘积的结果是一个形状为 (2, 2) 的矩阵。

注意，两个矩阵的维度需要满足乘法规则。具体而言，矩阵 `mat1` 的列数必须等于矩阵 `mat2` 的行数。在上述示例中，矩阵 `mat1` 的列数为 2，与矩阵 `mat2` 的行数相等。

如果提供了可选参数 `out`，则结果将存储在该参数指定的输出 Tensor 中。这样可以避免创建新的 Tensor 对象。
