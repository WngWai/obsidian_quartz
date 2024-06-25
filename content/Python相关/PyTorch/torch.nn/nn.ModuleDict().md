在PyTorch的`torch.nn`包中，`ModuleDict()`函数用于创建一个模块字典，用于存储和管理多个子模块，并使用键值对方式进行索引。
**函数定义**：
```python
torch.nn.ModuleDict(modules=None)
```

**参数**：
- `modules`（可选）：一个字典，其中键是子模块的名称，值是子模块本身。默认值为`None`，表示创建一个空的模块字典。

**示例**：
```python
import torch
import torch.nn as nn

# 创建一个空的模块字典
module_dict = nn.ModuleDict()

# 添加子模块
module_dict['linear'] = nn.Linear(10, 20)
module_dict['relu'] = nn.ReLU()

# 使用模块字典
input = torch.randn(32, 10)
output = module_dict['linear'](input)
output = module_dict['relu'](output)
```

在上述示例中，我们首先通过`nn.ModuleDict()`创建了一个空的模块字典 `module_dict`。

然后，我们使用键值对的方式向模块字典中添加了两个子模块。键是子模块的名称，值是子模块本身。我们添加了一个线性层 `nn.Linear(10, 20)`，其键为 `'linear'`，以及一个ReLU激活函数 `nn.ReLU()`，其键为 `'relu'`。

接下来，我们使用模块字典 `module_dict` 进行模型的前向传播。我们通过键值对的方式索引模块字典中的子模块，并将输入数据 `input`（大小为`(32, 10)`）传递给相应的子模块。

`ModuleDict`提供了一种方便的方式来管理和组织多个子模块，并使用键值对方式进行索引。与普通的Python字典不同，`ModuleDict`会自动跟踪和管理子模块的参数，并确保其在模型的训练和推理过程中得到正确的处理。通过使用键值对方式，可以更加直观地索引和使用子模块。

使用`ModuleDict`可以更好地组织和管理复杂的神经网络模型，提高代码的可读性和可维护性。