在 PyTorch 中，`retain_grad()` 是一个方法。它用于在计算图中**保留中间变量的梯度信息**。通常，在反向传播过程中，PyTorch 会**自动释放中间变量的梯度以节省内存**，但是通过 `retain_grad()` 可以保留这些中间变量的梯度信息，使得在计算图中进行更多的梯度计算。

以下是 `retain_grad()` 方法的基本信息：

**所属包：** torch

**定义：**
```python
retain_grad()
```

**参数介绍：**
该方法没有额外的参数。

**举例：**
```python
import torch

# 创建一个张量
x = torch.tensor([1.0], requires_grad=True)

# 执行一些操作
y = x * 2
z = y * 3

# 保留中间变量的梯度信息
y.retain_grad()
z.retain_grad()

# 计算损失
loss = z.sum()

# 执行反向传播
loss.backward()

# 打印梯度信息
print(x.grad)  # 梯度信息被保留
print(y.grad)  # 梯度信息被保留
print(z.grad)  # 梯度信息被保留
```

**输出：**
```
tensor([6.])
tensor([3.])
tensor([1.])
```

在上述示例中，`retain_grad()` 被用于保留中间变量 `y` 和 `z` 的梯度信息。这样，在计算损失后，可以通过这些变量的 `.grad` 属性访问它们的梯度信息。

需要注意的是，使用 `retain_grad()` 会增加内存消耗，因为梯度信息会被保留在计算图中，而不会在反向传播之后被释放。在大型模型或长序列中使用时，要注意内存的使用情况。