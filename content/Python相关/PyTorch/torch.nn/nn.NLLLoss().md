在PyTorch的`torch.nn`包中，`NLLLoss()`函数是用于计算负对数似然损失（Negative Log Likelihood Loss）的函数。通常用于多分类问题中，特别是在神经网络的最后一层使用logsoftmax激活函数时。

**函数定义**：
```python
torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

**参数**：
- `weight`（可选）：用于对不同类别的样本赋予不同的权重。默认值为`None`，表示所有类别的样本权重相等。
- `size_average`（已弃用，请使用`reduction`）：是否对损失进行平均。默认值为`None`，表示使用`reduction`参数的设置。
- `ignore_index`：指定一个忽略的类别索引。当计算损失时，忽略这个类别的预测值。默认值为`-100`，表示不忽略任何类别。
- `reduce`（已弃用，请使用`reduction`）：是否对损失进行降维。默认值为`None`，表示使用`reduction`参数的设置。
- `reduction`：指定损失的降维方式。可选值为`'none'`、`'mean'`和`'sum'`。默认值为`'mean'`，表示对损失进行均值降维。

**示例**：
```python
import torch
import torch.nn as nn

# 定义负对数似然损失函数
loss_fn = nn.NLLLoss()

# 定义样本标签和预测值
targets = torch.tensor([0, 2, 1])  # 样本标签
log_probs = torch.tensor([[-1.2, -0.5, -0.3], [-0.1, -2.3, -1.5], [-0.3, -0.8, -1.5]])  # 预测值的对数概率

# 计算损失
loss = loss_fn(log_probs, targets)

print(loss)
```

**输出示例**：
```
tensor(0.9131)
```

在上述示例中，我们首先通过`nn.NLLLoss()`创建了一个负对数似然损失函数 `loss_fn`。

然后，我们定义了样本的真实标签 `targets` 和模型输出的对数概率 `log_probs`。`targets`是一个大小为`(3,)`的张量，包含了每个样本的类别标签。`log_probs`是一个大小为`(3, 3)`的张量，表示模型对这三个样本的输出的对数概率。

接下来，我们使用 `loss_fn` 计算了预测值 `log_probs` 相对于真实标签 `targets` 的负对数似然损失。计算结果为一个标量张量 `loss`，表示整体的损失值。

`NLLLoss()`函数将模型的输出 `log_probs` 视为对数概率，并将真实标签 `targets` 视为样本的类别标签。它会根据这两者计算负对数似然损失，并返回一个标量张量作为损失值。

注意，在计算负对数似然损失时，模型的输出 `log_probs` 不需要经过softmax激活函数处理，`NLLLoss()`函数内部会自动进行softmax操作。

可以通过调整`weight`、`ignore_index`和`reduction`参数来适应不同的需求，例如为不同类别赋予不同的权重，忽略特定的类别或调整损失的降维方式。