在PyTorch的`torch.optim`包中，`RMSprop()`函数用于实现RMSprop优化算法。
**函数定义**：
```python
torch.optim.RMSprop(params, lr=<required parameter>, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```
**参数**：
- `params`（必需）：要进行优化的参数（张量）的迭代器或可训练模型的参数。
- `lr`（必需）：学习率（learning rate）。
- `alpha`（可选）：RMSprop的平滑系数。默认值为`0.99`。
- `eps`（可选）：用于数值稳定性的小常数。默认值为`1e-08`。
- `weight_decay`（可选）：权重衰减（weight decay）参数（L2正则化）。默认值为`0`。
- `momentum`（可选）：动量（momentum）因子。默认值为`0`。
- `centered`（可选）：是否使用RMSprop的中心化版本。默认值为`False`。

**示例**：
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型参数
params = nn.Linear(10, 5).parameters()

# 创建RMSprop优化器
optimizer = optim.RMSprop(params, lr=0.001)

# 在训练循环中使用优化器
for input, target in dataset:
    optimizer.zero_grad()  # 清零梯度
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
```

在上述示例中，我们首先通过`nn.Linear(10, 5).parameters()`创建了一个模型参数 `params`，这里使用了一个线性层作为示例。

然后，我们使用`optim.RMSprop()`函数创建了一个RMSprop优化器 `optimizer`。在这里，我们传递了模型参数 `params` 和学习率 `lr=0.001` 作为必需参数。

在训练循环中，我们首先调用 `optimizer.zero_grad()` 将梯度清零，然后进行前向传播、计算损失、反向传播和参数更新等操作。最后，我们调用 `optimizer.step()` 来更新模型参数。

RMSprop优化算法是一种自适应学习率的优化算法，它使用了平滑系数 `alpha` 和动量因子 `momentum` 来调整自适应学习率的更新方式。除了学习率 `lr` 这个必需参数外，还可以通过调整 `alpha`、`eps`、`weight_decay`、`momentum` 和 `centered` 等可选参数来控制优化算法的行为，以提高训练的效果和稳定性。