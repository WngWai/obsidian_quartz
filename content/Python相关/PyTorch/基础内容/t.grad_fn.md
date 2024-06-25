`grad_fn`属性在 PyTorch 中正式称为**梯度函数**（Gradient Function）。每个张量（Tensor）都有一个与之相关联的梯度函数，它记录了**张量是如何被创建的**。当你进行反向传播（backpropagation）计算梯度时，PyTorch 使用这些梯度函数来**构建计算图，然后计算梯度**。
由张量求得的**标量**，或者再由标量求得新的标量，都可以保留梯度函数，在反向求导中都成为计算图的一部分。
![[Pasted image 20231219231920.png]]

反向梯度函数：只要张量requires_grad打开了，除了叶张量，其他张量都有grad_fn属性。
grad_fn=\<MulBackward0\>**乘法**的反向梯度函数
grad_fn=\<AddBackward0\>**加法**




![[Pasted image 20231213152329.png]]

mul，y通过x的点积运算得来！
x.grad_fn是没有结果的，因为x是手动创建的，而非通过其他张量运算得来！
通过张量x生成新的变量y，y会继续为张量，且具有grad_fn属性，理解为存储与x的函数关系！
```python
z = y + 1

z.grad_fn

# 输出
<AddBackwrad0 at ........>
```
z同样是可微分的张量，且grad_fn属性中保留了函数关系，z是通过y+1的方式得到的。
`在可微张量创建的过程中，相关创建过程被保留到新张量的grad_fn属性中，实现系统的追踪`

在 PyTorch 中，`grad_fn` 是张量（`tensor`）的一个属性，用于跟踪创建张量的操作。这个属性指向一个表示张量的梯度计算图中操作的对象，该对象存储有关如何计算张量的梯度的信息。

具体来说，`grad_fn` 是一个指向 `Function` 对象的引用，`Function` 是 PyTorch 中用于定义操作和构建计算图的基本元素。每个张量都是通过一系列的操作创建的，而每个操作都对应着一个 `Function` 对象。这样，`grad_fn` 提供了构建计算图的信息，用于反向传播和梯度计算。

以下是一个简单的示例：

```python
import torch

# 创建张量
x = torch.tensor([1.0], requires_grad=True)

# 执行一些操作，创建新的张量
y = x * 2
z = y + 3

# 查看每个张量的 grad_fn 属性
print(x.grad_fn)  # None，因为 x 是用户创建的
print(y.grad_fn)  # <torch.autograd.function.MulBackward0 object at 0x...>
print(z.grad_fn)  # <torch.autograd.function.AddBackward0 object at 0x...>
```

在这个示例中，`x` 是用户创建的，因此它的 `grad_fn` 是 `None`。而 `y` 和 `z` 是通过操作创建的，它们的 `grad_fn` 分别指向 `MulBackward0` 和 `AddBackward0`，这些是 PyTorch 内部用于记录乘法和加法操作的 `Function` 类。

`grad_fn` 属性是 PyTorch 实现自动求导的关键部分，它使得 PyTorch 能够跟踪操作历史并执行反向传播，计算梯度。