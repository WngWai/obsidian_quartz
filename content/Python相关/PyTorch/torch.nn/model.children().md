在 PyTorch 中，`model.children()` 是一个用于获取模型中所有直接子模块的生成器方法。它返回一个生成器，该生成器递归地遍历模型的直接子模块，而**不包括模型本身**。

**所属包：** torch.nn.Module

**定义：**
`children()` 是 `torch.nn.Module` 类的方法，返回一个生成器，用于遍历模型的直接子模块。

**参数介绍：**
该方法没有额外的参数。

**举例：**
假设你有一个简单的神经网络模型，如下所示：

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# 创建模型实例
model = SimpleModel()

# 使用 children() 遍历模型的直接子模块
for child in model.children():
    print(child)
```

**输出：**
```
Linear(in_features=10, out_features=5, bias=True)
Linear(in_features=5, out_features=2, bias=True)
```

在上述示例中，`model.children()` 返回一个生成器，遍历了 `SimpleModel` 模型的两个线性层。与 `modules()` 不同，`children()` 只返回模型的直接子模块，而不递归地返回所有子模块，这使得它更适合查看模型的顶层结构。

这个方法在需要查看模型的顶层结构、仅对直接子模块进行某些操作时很有用。



