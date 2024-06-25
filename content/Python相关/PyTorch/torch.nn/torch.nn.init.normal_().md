在 PyTorch 中，`torch.nn.init.normal_()` 函数用于使用**正态分布初始化张量的值**。它是 PyTorch 初始化模块 `torch.nn.init` 中的一个函数，用于对模型的参数进行初始化。

以下是 `torch.nn.init.normal_()` 函数的基本信息：

**所属包：** torch.nn.init

**定义：**
```python
torch.nn.init.normal_(tensor, mean=0, std=1)
```

**参数介绍：**
- `tensor`：要初始化的张量。
- `mean`：正态分布的均值，默认为0。
- `std`：正态分布的标准差，默认为1。

**功能：**
使用正态分布（高斯分布）在均值为 `mean`，标准差为 `std` 的范围内初始化输入的张量。

**举例：**
```python
import torch
import torch.nn.init as init

# 创建一个3x3的张量
tensor = torch.empty(3, 3)

# 使用正态分布初始化张量
init.normal_(tensor, mean=0, std=0.1)

# 打印初始化后的张量
print(tensor)
```

**输出：**
```
tensor([[ 0.0363, -0.1301,  0.1338],
        [-0.0184,  0.0402, -0.0677],
        [ 0.0126, -0.0721, -0.0132]])
```

在上述示例中，`init.normal_()` 方法被用于在均值为0、标准差为0.1的正态分布中初始化一个3x3的张量。你可以在构建神经网络时，使用这个初始化方法来初始化权重或其他模型参数。