在 PyTorch 中，`cumsum()` 是一个 Tensor 对象的方法，用于计算 Tensor 中元素的**累积和**。它可以对整个 Tensor 进行累积求和，也可以沿着指定的维度进行累积求和。
```python
 cumsum(dim=None)
```
  - `dim`：可选参数，用于指定沿着哪个维度进行累积求和。默认情况下，对所有维度进行累积求和。

下面是一些示例：
1. 对整个 Tensor 进行累积求和：
   ```python
   import torch

   x = torch.tensor([1, 2, 3, 4, 5])
   cumsum = x.cumsum()
   print(cumsum)
   # 输出: tensor([ 1,  3,  6, 10, 15])
   ```
在这个例子中，`cumsum()` 方法对整个 Tensor 进行累积求和，返回一个新的 Tensor，其中每个元素是原始 Tensor 中对应位置之前所有元素的和。

2. 沿着指定的维度进行累积求和：

   ```python
   import torch

   x = torch.tensor([[1, 2, 3], [4, 5, 6]])
   cumsum_dim1 = x.cumsum(dim=1)
   print(cumsum_dim1)
   # 输出: tensor([[ 1,  3,  6],
   #              [ 4,  9, 15]])
   ```
在这个例子中，`cumsum(dim=1)` 对输入的二维 Tensor 沿着第一个维度（行）进行累积求和，返回一个新的二维 Tensor，其中每个元素是原始 Tensor 中对应位置之前该行所有元素的和。

`cumsum()` 方法返回一个新的 Tensor 对象，其中包含了累积求和的结果。根据参数的设置，可以对整个 Tensor 进行累积求和或沿着指定的维度进行累积求和。
