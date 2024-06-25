在 PyTorch 中，`torch.nn.Module` 是一个重要的数据类型，用于**定义神经网络模型**。`torch.nn.Module` 是所有**神经网络模型**的**基类**，你可以通过继承 `torch.nn.Module` 来创建自定义的神经网络模型。

定义网络图中每层的函数，以及模型的数据是如何前向传播的，前向传播中涉及其他函数也可以。
但不建议这么做，因为在前向传播中定义层。尽管这种方法在技术上是可行的，但它违反了**面向对象编程的封装原则**。这意味着**你的层实际上是在每次前向传播时被定义的**，这可能导致代码**难以理解和维护**。虽然可以在 `nn.Module` 的子类的 `__init__` 方法中定义层，但这并不是强制性的。如果你选择在前向传播中定义层，那么你可以这样做，但如上所述，这通常不是最佳实践。

`torch.nn.Module` 类提供了一些常用的功能，包括：
![[Pasted image 20231026130234.png]]

![[Pasted image 20231026130257.png]]
1. 参数管理：`torch.nn.Module` 可以管理模型的可学习参数（learnable parameters），例如权重和偏置项。你可以通过 `parameters()` 方法来访问模型的所有参数，并通过 `state_dict()` 方法来获取和设置模型的参数状态。
2. 前向传播：你可以重写 `torch.nn.Module` 类的 `forward()` 方法来定义模型的前向传播逻辑。在 `forward()` 方法中，你可以描述输入数据在网络中的流动，并通过各种层和操作来处理输入数据。
3. 模型组合和嵌套：`torch.nn.Module` 允许你将多个模型组合在一起，形成更复杂的网络结构。你可以使用 `nn.Module` 的子类作为模块的成员变量，并在 `forward()` 方法中调用这些子模块。

```python
import torch
from torch import nn
form torch.nn import functional as F

class LeNet(nn.Module):
 # 子模块创建，各个部分的初始化
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)
 
 # 子模块拼接，数据流向定义
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```

classes分类问题的**类别总数**！在定义类的实例对象时作为一个参数传入！

下面是一个示例，展示如何创建一个简单的自定义神经网络模型，继承自 `torch.nn.Module` 类：
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x

# 创建模型实例
model = SimpleNet()

# 打印模型结构
print(model)

# 输出
SimpleNet( 
(fc1): Linear(in_features=10, out_features=20, bias=True) (fc2): Linear(in_features=20, out_features=10, bias=True) )
```

在上述示例中，我们定义了一个名为 `SimpleNet` 的简单神经网络模型，它包含两个全连接层 (`nn.Linear`)。我们通过继承 `nn.Module` 类来创建模型类，并在 `__init__()` 方法中**定义模型的层**。在 `forward()` 方法中，我们定义了输入数据的**前向传播过程**。

通过实例化 `SimpleNet` 类，我们创建了一个模型对象 `model`。最后，我们打印模型对象，可以看到模型的结构信息。

## （重要）Module类对象的属性和方法
在 PyTorch 中，`model`（模型）是一个继承自 `nn.Module` 的自定义**类对象**。它可以包含各种属性（成员变量）和方法（成员函数），用于定义模型的结构、操作和行为。下面是一些常见的模型

可学习参数在线性层中指W、b

属性：
net.fc1 访问第几层
net.fc1.bias **带有梯度属性的b张量值**
net.fc1.bias.data
net.fc2.bias.data 只取张量值，而不管requires_grad=True梯度内容
net.fc2.weight.grad 访问w的梯度值

模型计算（Model computation）：
[[model.forward()]]定义模型的前向传播过程。当调用 `model(input)` 时，实际上是调用了 `model.forward(input)`。

模型参数（Parameters）：
[[model.parameters()]]返回模型中所有可学习参数的迭代器。这些参数是模型中需要进行优化和训练的权重和偏置项。
model.named_parameters() 返回所有可学习参数的详情


子模块（Submodules）：
[[model.modules()]]返回模型中所有的子模块（包括嵌套的子模块）的迭代器。可以用于遍历模型的所有子模块。

[[model.children()]]返回模型直接子模块的迭代器，但不含模型本身，不含嵌套的子模块？

模型状态（Model state）
[[model.state_dict()]]：返回一个字典，其中包含模型的**所有可学习参数**（如w,b）和缓冲区的当前状态。可以用于保存和加载模型的状态。model.state_dict()['2.bias'].data 通过名字，访问？？？
[[model.load_state_dict()]]加载预训练的模型状态字典，将模型的参数和缓冲区设置为给定状态。


[[model.train()]]将模型设置为**训练模式**，修改模型参数，启用训练相关的特性，例如 dropout 和 batch normalization。

[[model.eval()]]将模型设置为**评估模式**（测试状态），不会修改参数，禁用训练相关的特性，例如 dropout 和 batch normalization。


## 常见的成员变量命名方式
fc(fully connected layer)
全连接层
```python
self.fc1 = nn.Linear(in_features, out_features)
```

conv
卷积层
```python
self.conv1 = nn.Conv2d(3, 6, 5)
```

pool
池化层
```python
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
```

relu
激活函数层
```python
self.relu1 = nn.ReLU(inplace=True)
```


## super(LeNet, self).__init__()什么意思？
具体地说，`super` 的语法是 `super(<子类>, <对象>)`，其中 `<子类>` 是子类的类名，`<对象>` 是子类的对象。

在`nn.Module`的子类中，`__init__`方法是用来**初始化模块的**，而`super(LeNet, self).__init__()`的作用是**调用父类**（即`nn.Module`类）的`__init__`方法，**确保父类中定义的初始化逻辑得以执行**。

具体而言，`super(LeNet, self).__init__()`的参数`LeNet`表示当前类，`self`表示当前类的实例。这行代码调用了`nn.Module`类的`__init__`方法，以便在子类的初始化过程中执行一些父类中定义的必要的初始化工作。这样做是为了保证子类继承了父类的一些属性和方法，确保模块的正常运行。

在构建神经网络时，通常我们会定义自己的神经网络模型类，这个类需要继承自`nn.Module`。在子类的`__init__`方法中，我们经常需要添加一些自定义的属性和层，同时也需要调用父类的`__init__`方法来完成一些通用的初始化工作。这就是为什么在神经网络模型的`__init__`方法中经常看到`super(ClassName, self).__init__()`的原因。

## 引用容器中的模型模块（模型索引）
`nn.Module`是PyTorch中所有网络层和模型的基类。通过继承`nn.Module`，你可以创建一个复杂的网络架构，并且可以更精细地控制每一层的行为和属性。在自定义`nn.Module`中，你需要定义一个`__init__`方法来初始化网络层，并且定义一个`forward`方法来指定数据如何流经这些层。在自定义的模型中，你可以通过你为它们定义的属性名来访问层。

```python
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

net = CustomNet()

# 访问第一层
first_layer = net.fc1
```

在这个示例中，你不**能简单**地使用`net[0]`来访问第一层，因为`net`是一个自定义的`nn.Module`对象，其层是作为类的属性定义的。所以，你需要使用属性名（在这个例子中是`fc1`）来访问特定的层。