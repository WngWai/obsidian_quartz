在 PyTorch 中，`torch.nn.init.uniform_()` 函数用于使用**均匀分布初始化张量的值**。它是 PyTorch 初始化模块 `torch.nn.init` 中的一个函数，用于对模型的参数进行初始化。

以下是 `torch.nn.init.uniform_()` 函数的基本信息：

**所属包：** torch.nn.init

**定义：**
```python
torch.nn.init.uniform_(tensor, a=0, b=1)
```

**参数介绍：**
- `tensor`：要初始化的张量。
- `a`：均匀分布的下界，默认为0。
- `b`：均匀分布的上界，默认为1。

**功能：**
使用均匀分布在范围 `[a, b)` 内初始化输入的张量。

**举例：**
```python
import torch
import torch.nn.init as init

# 创建一个3x3的张量
tensor = torch.empty(3, 3)

# 使用均匀分布初始化张量
init.uniform_(tensor, a=-1, b=1)

# 打印初始化后的张量
print(tensor)
```

**输出：**
```
tensor([[ 0.1358,  0.6496,  0.7436],
        [-0.8377, -0.9364, -0.7856],
        [ 0.9562,  0.8372, -0.7625]])
```

在上述示例中，`init.uniform_()` 方法被用于在范围 `[-1, 1)` 内均匀分布地初始化一个3x3的张量。你可以在构建神经网络时，使用这个初始化方法来初始化权重或其他模型参数。