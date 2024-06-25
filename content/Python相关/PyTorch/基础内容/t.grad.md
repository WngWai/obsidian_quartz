在 PyTorch 中，`t.grad` 是一个属性，用于获取张量 `t` 的梯度值。它不接受任何参数，可以直接通过 `t.grad` 来**获取梯度值**。
![[Pasted image 20231026073016.png]]
先对谁求导，用x.grad而非z.grad查看梯度值！

要注意的是，`t.grad` 只在调用了 `backward()` 函数后才会有值，而且只有具有 `requires_grad=True` 属性的张量才会有梯度值。

果您确实希望为非叶张量填充 .grad 字段，请在非叶张量上使用 .retain_grad() 。

下面是一个示例，以展示如何使用 `t.grad` 属性来获取张量的梯度值：

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.sum()

# 计算 y 关于 x 的梯度
y.backward()

# 获取 x 的梯度值
grad = x.grad

# 输出 x 的梯度值
print(grad)  # 输出: tensor([1., 1.])
```

在以上示例中，我们创建了一个张量 `x`，并将其 `requires_grad` 属性设置为 True，表示需要计算梯度。然后，我们通过对 `x` 求和得到了张量 `y`。接着，我们调用 `y.backward()` 计算 `y` 关于 `x` 的梯度。最后，我们通过 `x.grad` 获取 `x` 的梯度值，并将其赋给变量 `grad`。

需要注意的是，`t.grad` 只能在执行了 `backward()` 函数后才有意义。如果没有调用 `backward()`，或者调用了 `backward()` 后未保留计算图（`retain_graph=False`），`t.grad` 的值将为 None。

总之，`t.grad` 是一个用于获取张量梯度的属性，可以用于访问计算图中的梯度值。通过调用 `backward()` 函数计算梯度后，可以使用 `t.grad` 来获取张量的梯度值，并进行进一步的梯度操作或参数更新。