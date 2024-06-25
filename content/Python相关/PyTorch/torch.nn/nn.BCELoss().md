在 PyTorch 中，`torch.nn.BCELoss()` 是**二分类交叉熵损失（Binary Cross Entropy Loss）** 的实现。它用于衡量二分类问题中模型输出与目标之间的差异，是训练二分类神经网络时常用的损失函数。

![[Pasted image 20240425092031.png|400]]

**定义：**
```python
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

**参数介绍：**
- `weight`：各个批次的损失之前的可选元素的权重。

- `size_average`：已过时，被 `reduce` 代替。如果为 `True`，则**损失会在批次中平均**。默认为 `True`。

- `reduce`：是否**对批次中的损失进行平均**。如果为 `True`，则返回的是平均损失，如果为 `False`，则返回的是**总损失**。默认为 `True`。

- `reduction`：指定损失的计算方式，可选值为 'none'、'mean' 或 'sum'。默认为 'mean'。

**功能：**
计算二分类交叉熵损失。

**举例：**
```python
import torch
import torch.nn as nn

# 创建一个 BCELoss 对象
criterion = nn.BCELoss()

# 模型输出（假设是模型的最终输出）
output = torch.sigmoid(torch.randn(3, requires_grad=True))

# 目标标签（假设是真实标签）
target = torch.empty(3).random_(2)

# 计算损失
loss = criterion(output, target)

# 打印损失
print(loss)
```

**输出：**
```
tensor(0.9836, grad_fn=<BinaryCrossEntropyBackward>)
```

在上述示例中，我们创建了一个 `BCELoss` 对象，计算了一个模型的**输出 `output` 和目标标签 `target` 之间的二分类交叉熵损失**。损失值是根据模型的输出和目标标签计算得到的。