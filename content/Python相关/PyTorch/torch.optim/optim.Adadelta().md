在PyTorch的`torch.optim`模块中，`Adadelta()`函数用于创建一个Adadelta优化器对象，该优化器使用Adadelta算法来更新模型的参数。
**函数定义**：
```python
torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

**参数**：
- `params`（必需）：一个可迭代的参数列表或参数字典，这些参数将由优化器进行更新。
- `lr`（可选）：学习率（缩放因子）的初始值。默认值为`1.0`。
- `rho`（可选）：Adadelta算法中的衰减因子，控制历史梯度平方的衰减速度。默认值为`0.9`。
- `eps`（可选）：用于数值稳定性的小值。默认值为`1e-6`。
- `weight_decay`（可选）：L2正则化（权重衰减）的权重。默认值为`0`，表示不使用权重衰减。

**示例**：
```python
import torch
import torch.optim as optim

# 定义模型参数
params = model.parameters()

# 创建一个Adadelta优化器
optimizer = optim.Adadelta(params, lr=1.0)

# 在训练循环中使用优化器更新模型参数
for inputs, labels in dataloader:
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()
```

在上述示例中，我们首先定义了模型的参数 `params`，可以通过`model.parameters()`获取。

然后，我们使用`optim.Adadelta()`创建了一个Adadelta优化器 `optimizer`，其中设置了学习率（缩放因子）为`1.0`。

在训练循环中，我们首先进行模型的前向传播和计算损失。然后，通过调用`optimizer.zero_grad()`来清零梯度，然后通过`loss.backward()`进行反向传播计算梯度。最后，通过调用`optimizer.step()`来更新模型的参数，优化器将使用Adadelta算法根据梯度信息来更新参数。

Adadelta是一种自适应学习率优化算法，它根据参数梯度的历史信息来自适应地调整学习率。它适用于训练深度神经网络，能够有效地处理非平稳目标函数和稀疏梯度的问题。