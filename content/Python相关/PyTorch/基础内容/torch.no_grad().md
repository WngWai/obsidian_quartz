`torch.no_grad()` 是一个上下文管理器（Context Manager）。作为范围就是with...：包裹的**代码块**，想想函数块！别整什么上下文了，有歧义！

更方便的是[[t.detach()]]，都相当于重新创建了一个叶张量（只是不可微，requires_grad关闭了）！

在这个上下文中，PyTorch 不会跟**踪张量的梯度信息**，从而减少内存消耗并加快代码的执行速度。相当于`张量的requires_grad被关了`，阻止了张量的autograd功能
![[Pasted image 20231213223104.png]]
直接通过前向传播，得到z的结果。

使用 `torch.no_grad()` 的常见场景是在推理阶段或验证阶段，当我们不需要计算梯度时。在这些阶段，我们通常只是对模型进行前向传播（**就是通过计算关系，从叶张量出发，得到输出张量**），而不需要反向传播和参数更新，因此可以通过禁用梯度计算来提高性能和减少内存消耗。

以下是使用 `torch.no_grad()` 的示例：

```python
import torch

# 定义模型
model = torch.nn.Linear(10, 1)

# 输入数据
input = torch.randn(32, 10)

# 在推理阶段禁用梯度计算
with torch.no_grad():
    # 前向传播
    output = model(input)

# 在禁用梯度计算的上下文中，output 不会保留梯度信息
print(output.requires_grad)  # 输出: False
```

在上述示例中，我们定义了一个简单的线性模型 `model`，并随机生成一个输入张量 `input`。然后，我们使用 `torch.no_grad()` 创建了一个上下文，在该上下文中执行了模型的前向传播操作。在这个上下文中，`output` 张量不会保留梯度信息，因为梯度计算被禁用。最后，我们打印了 `output` 张量的 `requires_grad` 属性，确认它不需要梯度。

通过使用 `torch.no_grad()` 上下文管理器，我们可以**明确地控制梯度计算的开启和关闭**，以便在合适的时候提高代码的性能和减少内存消耗。