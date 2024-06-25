在PyTorch中，`nn.Sequential`函数是神经网络模块`torch.nn`中的一个类，用于**按顺序组合多个神经网络模块**。`nn.Sequential`允许将这些模块按照顺序堆叠起来，以构建更复杂的神经网络模型。可以理解为一个封装模块的容器。
它可以接收一个子模块的**有序字典**(OrderedDict) 或者**一系列子模块**作为参数来逐一添加 `Module` 的实例，⽽模型的**前向计算就是将这些实例按添加的顺序逐⼀计算**。

**类定义**：
```python
class torch.nn.Sequential(*args)
```
**参数**：
- `*args`：神经网络模块，按顺序传递给`nn.Sequential`构造函数。**每个模块都是一个独立的参数**，可以是`nn.Module`的子类实例对象。相当于嵌套了model神经网络模型

**方法**：
- `forward(input)`：定义了模块的**前向传播**逻辑。

**示例**：
```python
import torch
import torch.nn as nn

# 创建一个简单的神经网络模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)

# 打印模型结构
print(model)

# 定义输入张量
input_tensor = torch.randn(100, 784)

# 进行前向传播
output_tensor = model(input_tensor)

print(output_tensor.shape)
```

**输出示例**：
```python
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
  (3): Softmax(dim=1)
)
torch.Size([100, 10])
```

在上述示例中，我们首先创建了一个简单的神经网络模型 `model`，它由多个神经网络模块按顺序组成。具体地，我们使用`nn.Sequential`将四个模块按顺序堆叠起来。

这个模型的结构如下：
1. 线性变换层 `nn.Linear(784, 256)`，将输入特征的大小从 784 转换为 256。
2. ReLU 激活函数 `nn.ReLU()`，对线性变换的结果进行非线性处理。
3. 线性变换层 `nn.Linear(256, 10)`，将输入特征的大小从 256 转换为 10。
4. Softmax 激活函数 `nn.Softmax(dim=1)`，将输出转换为概率分布。

然后，我们打印了模型的结构，可以看到模型中每个模块的名称和类型。

接下来，我们定义了一个输入张量 `input_tensor`，其中包含 100 个样本，每个样本的特征大小为 784。

最后，我们使用 `model` 对输入张量进行前向传播，得到输出张量 `output_tensor`。输出张量的形状为 `(100, 10)`，其中 100 是样本数量，10 是输出特征的大小。

`nn.Sequential`函数在神经网络中用于按顺序组合多个模块，方便快捷地构建神经网络模型。通过将不同类型的模块按照顺序传递给`nn.Sequential`构造函数，可以定义不同层之间的连接关系，并实现复杂的神经网络结构。

### 引用容器中的模型模块（模型索引）

```python
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```
net[0]：访问nn神经网络的第一层，这里的第一层指线性回归模型；
.weight.data范围W属性的底层数据；
.normal_：对W的数据进行正态分布的随机初始化。

`nn.Sequential`是一个有序的容器，网络层会按照你添加它们的顺序被添加到容器中。使用这个容器的好处是它允许你快速地创建网络，而不需要定义一个自定义的`nn.Module`类。`nn.Sequential`模型的层可以通过索引访问，就像访问普通的Python列表一样。

```python
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 访问第一层
first_layer = net[0]
```


### model.parameters()或net.parameters()
在给定的代码中，`net`是一个`nn.Sequential`对象，它代表一个顺序容器，用于按顺序组合多个神经网络层。
`net.parameters()`是一个方法调用，用于获取`net`中所有可学习参数的**迭代器**。可学习参数指的是模型中需要通过训练来更新的参数，如**权重w和偏置b**。
具体而言，`net.parameters()`返回一个迭代器，可以用于遍历`net`中所有的可学习参数。每个可学习参数都是一个`Parameter`对象，它包含了参数的张量数据以及梯度信息。
这个迭代器通常用于将可学习参数传递给优化器（如随机梯度下降算法），以便进行参数更新。通过遍历这个迭代器，可以获取模型中所有需要更新的参数，并将其传递给优化器的参数更新函数。
总结来说，`net.parameters()`用于获取模型中所有可学习参数的迭代器，以便在训练过程中更新这些参数。


### 直接排序和有序字典
直接排序
```python
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)

#
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```

有序字典OrderedDict
```python
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net2)

#
Sequential(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)
```