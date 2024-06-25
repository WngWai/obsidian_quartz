在 PyTorch 中，`t.backward()` 是用于计算张量 `t` 关于自身的**梯度**的函数。它会自动计算 `t` 的梯度，并将梯度值累加到 `t.grad` 属性中。`t.backward()` 函数没有参数，它的调用方式如下：

在一张计算图上**只能执行一次**反向传播，不管是对中间节点还是输出节点执行，执行完这张计算图的**其他节点就不能再次执行反向传播了**。
且为了节省资源，默认下中间变量的梯度是没有保存的！

```python
t.backward(gradient=None, retain_graph=None, create_graph=False)
```

- `gradient`：可选参数，用于指定**梯度的初始值**。默认为 None，表示使用**标量 1** 作为初始梯度值。

- `retain_graph`：可选参数，用于指定是否保留计算图。默认为 None，表示根据需要自动释放计算图。

- `create_graph`：可选参数，用于指定是否在计算图中创建梯度图。默认为 False，表示不创建梯度图。

![[Pasted image 20231026073016.png]]

是对函数进行反向求导！Z对X进行求导！

下面是一个示例，以展示 `t.backward()` 函数的使用：

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x.sum()

# 计算 y 关于 x 的梯度
y.backward()

# 输出 x 的梯度值
print(x.grad)  # 输出: tensor([1., 1.])
```

在以上示例中，我们创建了一个张量 `x`，并将其 `requires_grad` 属性设置为 True，表示需要计算梯度。然后，我们通过对 `x` 求和得到了张量 `y`。接着，我们调用 `y.backward()` 计算 `y` 关于 `x` 的梯度，并将梯度值累加到 `x.grad` 属性中。最后，我们打印出 `x.grad`，即 `x` 的梯度值。

需要注意的是，为了使用 `backward()` 函数计算梯度，张量必须满足一些条件，如 `requires_grad=True`，并且该张量是一个标量（只有一个元素），或者是通过某个标量进行聚合计算得到的。此外，如果在计算图中使用了某个张量的梯度，而且希望保留计算图以供后续使用，可以将 `retain_graph` 参数设置为 True。

总之，`t.backward()` 是一个在 PyTorch 中用于计算梯度的重要函数，它可以根据链式法则自动计算张量的梯度，并将梯度值累加到 `grad` 属性中，方便进行反向传播和参数更新。