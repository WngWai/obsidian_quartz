在PyTorch中，`torch.randn()`函数用于创建一个从**标准正态分布（均值为0，标准差为1）中抽取**的随机张量。它接受一个或多个参数来指定张量的形状，并返回一个形状匹配的随机张量。

```python
torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

**参数**：
- `*size`：一个可变数量的参数，用于指定张量的形状。可以是整数值，元组或列表。例如，`torch.randn(2, 3)` 创建一个形状为 `(2, 3)` 的随机张量。
表现形式有torch.randn(size=())创建一个**标量**（scalar），而非空得标量；
torch.randn(())也可以这种形式，但就是不能什么都不写！torch.randn()

- `out`：可选参数，用于指定输出张量的位置。如果提供了该参数，函数会将结果存储在指定的张量中。
- `dtype`：可选参数，用于指定输出张量的数据类型。默认值为 `None`，表示使用默认的数据类型。
- `layout`：可选参数，用于指定输出张量的布局。默认值为 `torch.strided`。
- `device`：可选参数，用于指定输出张量所在的设备。默认值为 `None`，表示使用默认设备。
- `requires_grad`：可选参数，用于指定输出张量是否需要梯度计算。默认值为 `False`。

**示例**：
```python
import torch

# 创建一个形状为 (2, 3) 的随机张量
randn_tensor = torch.randn(2, 3)

# 查看张量的值和形状
print("随机张量：")
print(randn_tensor)
print("张量的形状：")
print(randn_tensor.shape)
```

**输出示例**：
```
随机张量：
tensor([[-0.2686,  0.1547, -0.2550],
        [ 0.7741,  1.0489, -0.3012]])
张量的形状：
torch.Size([2, 3])
```

在上述示例中，我们调用 `torch.randn(2, 3)` 来创建一个形状为 `(2, 3)` 的随机张量。函数返回的结果是一个形状为 `(2, 3)` 的随机张量，其中的值是从标准正态分布中抽取的。

然后，我们打印输出随机张量 `randn_tensor` 的值和形状。可以看到，张量的值是随机的，并且符合标准正态分布，形状为 `(2, 3)`。

`torch.randn()`函数在模型权重初始化、生成随机噪声等需要随机数的场景中非常有用。它可以帮助我们快速创建指定形状的随机张量。