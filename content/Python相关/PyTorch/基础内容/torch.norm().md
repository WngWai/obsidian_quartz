`torch.norm()` 是 PyTorch 中用于计算向量或矩阵的**范数**（norm，长度值）的函数。它可以计算向量的 Lp 范数和矩阵的各种范数。
```python
torch.norm(input, p='fro', dim=None, keepdim=False, out=None)
```

  - `input`：输入 Tensor，可以是向量或矩阵。
  - `p`：可选参数，用于指定要计算的范数类型。默认为 'fro'，表示计算矩阵的 Frobenius 范数。常见的选项包括：
    - `1`：计算向量的 L1 范数（**曼哈顿范数**）。
	>$\|\mathbf{x}\|_1=\sum_{i=1}^n|x_i|$

    - `2`：计算向量的 L2 范数（**欧几里得范数**）（默认）。
	>$\|\mathbf{x}\|_2=\sqrt{\sum_{i=1}^nx_i^2}$

    - `inf`：计算向量的无穷范数。
    - `-inf`：计算向量的负无穷范数。
    - `'nuc'`：计算矩阵的核范数。
  - `dim`：可选参数，用于指定计算范数的维度。默认为 `None`，表示计算整个 Tensor 的范数。
  - `keepdim`：可选参数，用于指定是否保持结果的维度。默认为 `False`，表示降维计算范数。
  - `out`：可选参数，用于指定输出 Tensor。

下面是一些示例：

1. 计算向量的 L2 范数：

   ```python
   import torch

   x = torch.tensor([1, 2, 3])
   norm = torch.norm(x, p=2)
   print(norm)
   # 输出: tensor(3.7417)
   ```

   在这个例子中，`torch.norm(x, p=2)` 计算向量 `x` 的 L2 范数，即欧几里德范数。

2. 计算矩阵的 Frobenius 范数：

   ```python
   import torch

   mat = torch.tensor([[1, 2], [3, 4], [5, 6]])
   norm = torch.norm(mat)
   print(norm)
   # 输出: tensor(9.5394)
   ```

   在这个例子中，`torch.norm(mat)` 计算矩阵 `mat` 的弗罗贝尼乌斯(Frobenius)范数。

3. 沿着指定维度计算矩阵的 L1 范数：

   ```python
   import torch

   mat = torch.tensor([[1, 2], [3, 4], [5, 6]])
   norm = torch.norm(mat, p=1, dim=1)
   print(norm)
   # 输出: tensor([3, 7, 11])
   ```

   在这个例子中，`torch.norm(mat, p=1, dim=1)` 沿着维度1（行）计算矩阵 `mat` 的 L1 范数。

`torch.norm()` 函数返回一个标量 Tensor，其中包含计算得到的范数结果。根据参数的设置，可以计算向量的 Lp 范数、矩阵的 Frobenius 范数或沿着指定维度计算矩阵的各种范数。