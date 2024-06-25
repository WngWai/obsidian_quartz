`torch.mv()` 是 PyTorch 中用于执行**矩阵和向量相乘**的函数。它接受一个二维矩阵和一个一维向量，并返回它们的矩阵-向量乘积结果。
以下是对 `torch.mv()` 函数的详细介绍和示例：
```python
torch.mv(mat, vec, out=None)
```

  - `mat`：二维输入矩阵，形状为 (M, N)。

  - `vec`：一维输入向量，形状为 (N,)。

  - `out`：可选参数，用于指定输出 Tensor。

下面是一个示例：
```python
import torch

mat = torch.tensor([[1, 2, 3], [4, 5, 6]])
vec = torch.tensor([7, 8, 9])

result = torch.mv(mat, vec)
print(result)
# 输出: tensor([ 50, 122])
```

在这个例子中，`torch.mv(mat, vec)` 对输入的矩阵 `mat` 和向量 `vec` 进行矩阵-向量乘积操作。矩阵 `mat` 的形状为 (2, 3)，向量 `vec` 的形状为 (3,)，因此乘积的结果是一个形状为 (2,) 的向量。

注意，矩阵和向量的维度需要满足乘法规则。具体而言，矩阵的列数必须等于向量的长度。在上述示例中，矩阵 `mat` 的列数为 3，与向量 `vec` 的长度相等。

如果提供了可选参数 `out`，则结果将存储在该参数指定的输出 Tensor 中。这样可以避免创建新的 Tensor 对象。

希望这个例子能够帮助您理解 `torch.mv()` 函数的用法。如果您有任何其他问题，请随时提问。