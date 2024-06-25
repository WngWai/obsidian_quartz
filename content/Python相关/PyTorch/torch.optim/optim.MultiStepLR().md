在PyTorch的`torch.optim`模块中，`MultiStepLR()`函数用于创建一个学习率调度器，该调度器在预定义的多个步骤上调整学习率。
**函数定义**：
```python
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)
```
**参数**：
- `optimizer`（必需）：优化器对象，例如`torch.optim.SGD`。
- `milestones`（必需）：一个包含预定义的步骤的列表。在这些步骤上，学习率将按照`gamma`进行调整。
- `gamma`（可选）：学习率缩放因子。学习率会在每个步骤后乘以`gamma`。默认值为`0.1`。
- `last_epoch`（可选）：上一个epoch的索引。默认值为`-1`，表示从头开始计算。
- `verbose`（可选）：是否打印调整学习率的详细信息。默认值为`False`。

**示例**：
```python
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 创建一个优化器（如SGD）
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建一个学习率调度器（在第30和第60个epoch将学习率缩小为原来的0.1）
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

# 在训练循环中更新学习率
for epoch in range(100):
    # 训练模型
    train()
    
    # 更新优化器的学习率
    scheduler.step()
```

在上述示例中，我们首先创建了一个优化器 `optimizer`，例如使用随机梯度下降（SGD）优化器，并设置学习率为`0.1`。

然后，我们使用`lr_scheduler.MultiStepLR()`创建了一个学习率调度器 `scheduler`，其中设置了两个里程碑（milestones），即第30个和第60个epoch。在这些里程碑上，学习率将被缩小为原来的`0.1`。

在训练循环中，我们首先进行模型的训练。然后，通过调用`scheduler.step()`来更新优化器的学习率。调度器会根据当前epoch和预定义的里程碑，自动调整学习率。

`MultiStepLR`学习率调度器允许我们在不同的epoch上按照预定义的方式调整学习率。通过在训练过程中灵活地选择里程碑，我们可以根据训练的进展情况来调整学习率，以获得更好的模型性能。这对于需要在不同训练阶段采用不同学习率的任务非常有用。