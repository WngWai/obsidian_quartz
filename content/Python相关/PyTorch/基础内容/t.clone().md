在 PyTorch 中，`clone()` 是一个 Tensor 对象的方法，用于创建一个与原始 Tensor 具有**相同数据和属性的新的独立副本**。`clone()` 方法通常用于创建在计算图中需要保留梯度信息的新 Tensor。
```python
clone(memory_format=None)
```
  - `memory_format`：可选参数，用于指定新 Tensor 的存储格式（内存布局）。

下面是一些示例：

1. 使用默认参数 `clone()` 创建一个新的 Tensor：
   ```python
   import torch

   x = torch.tensor([1, 2, 3])
   y = x.clone()
   print(y)
   # 输出: tensor([1, 2, 3])
   ```

   在这个例子中，`y` 是 `x` 的副本，它们具有相同的值和属性，但是它们是不同的 Tensor 对象。修改 `y` 不会影响 `x`，它们是独立的。

2. 使用 `clone()` 创建一个新的 Tensor，并指定存储格式：

   ```python
   import torch

   x = torch.tensor([1, 2, 3])
   y = x.clone(memory_format=torch.contiguous_format)
   print(y)
   # 输出: tensor([1, 2, 3], dtype=torch.int64)
   ```

   在这个例子中，通过传递 `memory_format=torch.contiguous_format` 参数，创建了一个具有连续内存布局的新 Tensor。

`clone()` 方法返回一个新的 Tensor 对象，该对象具有与原始 Tensor 相同的数据和属性。通过 `clone()` 创建的新 Tensor 是独立的，对其进行修改不会影响原始 Tensor。

需要注意的是，`clone()` 方法不会复制 Tensor 的梯度信息。如果需要复制梯度信息，可以使用 `clone().detach()` 或 `clone().detach().requires_grad_(True)`。
