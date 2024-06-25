在PyTorch的`torch.nn`包中，`Sigmoid()`函数用于实现Sigmoid激活函数。
输入特征映射到0到1之间的概率范围。

搞半天是将**回归函数+sogmoid函数=逻辑回归函数** 

**函数定义**：
```python
torch.nn.Sigmoid()
```

**参数**：
`Sigmoid()`函数没有额外的可调参数。

**示例**：
```python
import torch
import torch.nn as nn

# 定义Sigmoid激活函数
activation = nn.Sigmoid()

# 输入张量
input_tensor = torch.tensor([-1.0, 2.0, -3.0, 4.0])

# 应用Sigmoid激活函数
output_tensor = activation(input_tensor)

print(output_tensor)
```

**输出示例**：
```
tensor([0.2689, 0.8808, 0.0474, 0.9820])
```

在上述示例中，我们首先通过`nn.Sigmoid()`创建了一个Sigmoid激活函数 `activation`。

然后，我们定义了一个输入张量 `input_tensor`，其中包含了一些实数值。这个输入张量的大小可以是任意的。

接下来，我们使用 `activation` 应用Sigmoid激活函数到输入张量上，得到输出张量 `output_tensor`。

Sigmoid激活函数的作用是将输入张量中的每个元素压缩到0到1之间的范围内。在示例中，输入张量中的每个元素都经过了Sigmoid函数的计算，得到了相应的输出值。

`Sigmoid()`函数没有额外的可调参数，它的作用仅仅是将输入张量中的每个元素进行Sigmoid函数的计算。

Sigmoid激活函数常用于二分类问题或需要将输出值映射到0到1之间的任务中。它的输出值可以被解释为概率或置信度。

![[Pasted image 20240425091835.png|400]]
