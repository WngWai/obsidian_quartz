`torch.rand()` 是 PyTorch 中的一个函数，用于生成具有**均匀分布（在0到1之间）** 的随机数。这些随机数在需要随机但非正态分布的数值时很有用，比如在某些类型的模拟或初始化神经网络权重时（尽管通常更倾向于使用正态分布进行权重初始化）。

```python
python复制代码torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```


- `size`（可变参数）: 一个定义输出**张量形状**的整数序列（例如，要生成一个形状为 `(m, n)` 的二维张量，可以传入 `m, n`）。
- `out`（可选）: 输出张量。
- `dtype`（可选）: 所需的数据类型。默认为 `None`，在这种情况下，将使用全局默认数据类型（通常是 `torch.float32`）。
- `layout`（可选）: 布局类型。默认为 `torch.strided`。
- `device`（可选）: 所需设备（CPU 或 GPU）。默认为 `None`，这意味着它将使用当前设备。
- `requires_grad`（可选）: 设置为 `True` 如果创建的张量需要梯度。默认为 `False`。

### 返回值

返回一个具有从0到1（包括0但不包括1）的均匀分布的随机数填充的张量，其形状由 `size` 参数定义。

### 示例

1. 生成一个形状为 `(3, 4)` 的二维张量：

```python
python复制代码import torch    x = torch.rand(3, 4)  print(x)
```

2. 生成一个包含5个随机数的一维张量：

```python
python复制代码x = torch.rand(5)  print(x)
```

每次调用 `torch.rand()` 时，都会生成一组新的随机数。这些随机数在每次程序运行时都会不同，因为它们是从一个均匀分布中随机抽取的。

请注意，`torch.rand()` 和 `torch.randn()` 之间的主要区别在于它们所抽取的随机数所服从的分布不同：`torch.rand()` 是从均匀分布中抽取的，而 `torch.randn()` 是从标准正态分布中抽取的。