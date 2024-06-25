在PyTorch的`torch.nn`包中，`ModuleList()`函数用于创建一个模块列表，用于**存储和管理多个子模块**。
**函数定义**：
```python
torch.nn.ModuleList(modules=None)
```
**参数**：
- `modules`（可选）：一个可迭代的模块列表。默认值为`None`，表示创建一个空的模块列表。
**示例**：
```python
import torch
import torch.nn as nn

# 创建一个空的模块列表
module_list = nn.ModuleList()

# 添加子模块
module_list.append(nn.Linear(10, 20))
module_list.append(nn.ReLU())

# 使用模块列表
input = torch.randn(32, 10)
output = module_list[0](input)
output = module_list[1](output)
```

在上述示例中，我们首先通过`nn.ModuleList()`创建了一个空的模块列表 `module_list`。

然后，我们使用`append()`方法向模块列表中添加了两个子模块，分别是一个线性层 `nn.Linear(10, 20)` 和一个ReLU激活函数 `nn.ReLU()`。

接下来，我们使用模块列表 `module_list` 进行模型的前向传播。我们首先将输入数据 `input`（大小为`(32, 10)`）传递给模块列表中的第一个子模块 `module_list[0]`，然后将输出传递给第二个子模块 `module_list[1]`。

`ModuleList`提供了一种方便的方式来管理和组织多个子模块。与Python的普通列表不同，`ModuleList`会自动跟踪和管理子模块的参数，并确保其在模型的训练和推理过程中得到正确的处理。通过使用`append()`方法或类似的操作，可以在运行时动态地向模块列表中添加、删除或替换子模块。

使用`ModuleList`可以更好地组织和管理复杂的神经网络模型，提高代码的可读性和可维护性。