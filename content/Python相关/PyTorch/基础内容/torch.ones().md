在PyTorch中，`torch.ones()`函数用于创建一个**全一张量**（Tensor）。它接受一个或多个参数来指定张量的形状，并返回一个形状匹配的全一张量。
```python
torch.ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```
**参数**：
- `*size`：一个可变数量的参数，用于指定张量的形状。可以是整数值，元组或列表。例如，`torch.ones(2, 3)` 创建一个形状为 `(2, 3)` 的全一张量。
- `out`：可选参数，用于指定输出张量的位置。如果提供了该参数，函数会将结果存储在指定的张量中。
- `dtype`：可选参数，用于指定输出张量的数据类型。默认值为 `None`，表示使用默认的数据类型。
- `layout`：可选参数，用于指定输出张量的布局。默认值为 `torch.strided`。
- `device`：可选参数，用于指定输出张量所在的设备。默认值为 `None`，表示使用默认设备。
- `requires_grad`：可选参数，用于指定输出张量是否需要梯度计算。默认值为 `False`。
**示例**：
```python
import torch

# 创建一个形状为 (2, 3) 的全一张量
ones_tensor = torch.ones(2, 3)

# 查看张量的值和形状
print("全一张量：")
print(ones_tensor)
print("张量的形状：")
print(ones_tensor.shape)
```

**输出**：
```
全一张量：
tensor([[1., 1., 1.],
        [1., 1., 1.]])
张量的形状：
torch.Size([2, 3])
```

在上述示例中，我们调用 `torch.ones(2, 3)` 来创建一个形状为 `(2, 3)` 的全一张量。函数返回的结果是一个形状为 `(2, 3)` 的全一张量。

然后，我们打印输出全一张量 `ones_tensor` 的值和形状。可以看到，张量的值全部为1，形状为 `(2, 3)`。

`torch.ones()`函数在初始化模型参数、创建占位符张量等场景中非常有用。它可以帮助我们快速创建指定形状且值为1的张量。