在PyTorch的`torch.optim`包中，`Adam()`函数用于实现Adam优化算法。

**函数定义**：
```python
torch.optim.Adam(params, lr=<required parameter>, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

**参数**：
- `params`（必需）：要进行优化的参数（张量）的迭代器或可训练模型的参数。
- `lr`（必需）：学习率（learning rate）。
- `betas`（可选）：用于计算梯度和平方梯度的运行平均值的系数。默认值为`(0.9, 0.999)`。
- `eps`（可选）：用于数值稳定性的小常数。默认值为`1e-08`。

- `weight_decay`（可选）：权重衰减（weight decay）参数（L2正则化）。默认值为`0`。

如果`weight_decay`被设置为一个正数（例如0.001），则优化器会在每次更新模型权重时，将权重向量乘以`(1 - weight_decay)`，然后再应用梯度更新。这实际上等同于在损失函数中添加一个与权重向量的L2范数成比例的项，并乘以`weight_decay`。

- `amsgrad`（可选）：是否使用AMSGrad变体。默认值为`False`。

**示例**：
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型参数
params = nn.Linear(10, 5).parameters()

# 创建Adam优化器
optimizer = optim.Adam(params, lr=0.001)

# 在训练循环中使用优化器
for input, target in dataset:
    optimizer.zero_grad()  # 清零梯度
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
```

在上述示例中，我们首先通过`nn.Linear(10, 5).parameters()`创建了一个模型参数 `params`，这里使用了一个线性层作为示例。

然后，我们使用`optim.Adam()`函数创建了一个Adam优化器 `optimizer`。在这里，我们传递了模型参数 `params` 和学习率 `lr=0.001` 作为必需参数。

在训练循环中，我们首先调用 `optimizer.zero_grad()` 将梯度清零，然后进行前向传播、计算损失、反向传播和参数更新等操作。最后，我们调用 `optimizer.step()` 来更新模型参数。

Adam优化算法是一种自适应学习率的优化算法，结合了动量方法和RMSprop算法。除了学习率 `lr` 这个必需参数外，还可以通过调整 `betas`、`eps`、`weight_decay` 和 `amsgrad` 等可选参数来控制优化算法的行为，以提高训练的效果和稳定性。