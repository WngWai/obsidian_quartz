在Python的PyTorch库中，`model.load_state_dict()`函数用于加载模型的状态字典（state_dict），从而恢复模型的参数。
**函数定义**：
```python
model.load_state_dict(state_dict, strict=True)
```
**参数**：
以下是`model.load_state_dict()`函数中常用的参数：
- `state_dict`：要加载的模型状态字典。它是一个Python字典对象，其中键是模型的参数名称，值是对应的参数张量。
- `strict`（可选）：指定是否严格加载模型参数的标志。如果设置为`True`（默认值），则加载的状态字典必须与模型的参数名称和形状完全匹配。如果设置为`False`，则允许加载部分状态字典，忽略不匹配的键。注意，如果启用了严格模式，并且状态字典与模型不匹配，将引发`RuntimeError`。

**示例**：
以下是使用`model.load_state_dict()`函数加载模型状态字典的示例：

```python
import torch
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

# 创建模型实例
model = SimpleModel()

# 保存模型的状态字典
torch.save(model.state_dict(), "model_state_dict.pt")

# 创建一个新的模型实例
new_model = SimpleModel()

# 加载模型的状态字典到新模型中
new_model.load_state_dict(torch.load("model_state_dict.pt"))
```

在上述示例中，我们首先导入了`torch`和`torch.nn`库，并定义了一个简单的模型`SimpleModel`，它包含一个线性层。

然后，我们创建了一个模型实例`model`。

接下来，我们使用`torch.save()`函数将模型的状态字典保存到文件`model_state_dict.pt`中。

然后，我们创建了一个新的模型实例`new_model`。

最后，我们使用`new_model.load_state_dict()`函数加载保存的模型状态字典到新模型中，从而恢复了模型的参数。

通过运行上述代码，我们可以成功加载保存的模型状态字典，并将其应用于新模型中。

除了上述示例中的参数，`model.load_state_dict()`函数还有其他可用的参数和选项，用于更精细的控制加载状态字典的行为。详细的参数说明可以参考PyTorch官方文档。