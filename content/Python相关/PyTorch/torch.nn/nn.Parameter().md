在PyTorch中，`nn.Parameter()`函数用于**将张量包装成可训练的参数**。它通常在自定义模型的构造函数中使用，用于定义需要优化的模型参数。

**函数定义**：
```python
nn.Parameter(data=None, requires_grad=True)
```

**参数**：
以下是`nn.Parameter()`函数中的参数：

- `data`：可选参数，需要包装成可训练参数的张量。如果未提供该参数，则会创建一个空的张量。

- `requires_grad`：可选参数，指定是否计算张量的梯度。默认值为`True`，即计算梯度。

**示例**：
以下是使用`nn.Parameter()`函数创建可训练参数的示例：

```python
import torch
import torch.nn as nn

# 创建一个需要优化的张量
weights = torch.randn(10, 20)

# 将张量包装成可训练参数
weights_param = nn.Parameter(weights)

print(weights_param)  # 输出: Parameter containing: tensor([...])

# 使用可训练参数构建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weights = nn.Parameter(torch.randn(10, 20))
    
    def forward(self, x):
        output = torch.matmul(x, self.weights)
        return output

model = Model()
print(model.weights)  # 输出: Parameter containing: tensor([...])
```

在上述示例中，我们首先导入了所需的模块和类。

然后，我们创建了一个随机初始化的张量`weights`，其形状为`[10, 20]`。

接下来，我们使用`nn.Parameter()`函数将张量`weights`包装成可训练参数`weights_param`。
可训练参数`weights_param`可以像普通张量一样使用，但它会被自动跟踪并计算梯度。
在示例中，我们还展示了如何在自定义模型的构造函数中使用`nn.Parameter()`函数创建可训练参数。在`Model`类中，我们将`weights`作为可训练参数`self.weights`，并在模型的`forward()`方法中使用它。