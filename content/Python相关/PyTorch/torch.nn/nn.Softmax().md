在PyTorch的`torch.nn`包中，`Softmax()`函数用于实现Softmax激活函数。
**函数定义**：
```python
torch.nn.Softmax(dim=None)
```
**参数**：
- `dim`（可选）：指定Softmax操作的维度。默认值为`None`，表示在最后一个维度上进行Softmax操作。
**示例**：
```python
import torch
import torch.nn as nn

# 定义Softmax激活函数
activation = nn.Softmax(dim=1)

# 输入张量
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 应用Softmax激活函数
output_tensor = activation(input_tensor)

print(output_tensor)
```

**输出示例**：
```
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])
```

在上述示例中，我们首先通过`nn.Softmax()`创建了一个Softmax激活函数 `activation`。

然后，我们定义了一个输入张量 `input_tensor`，其中包含了一些实数值。这个输入张量的大小可以是任意的。

接下来，我们使用 `activation` 应用Softmax激活函数到输入张量上，得到输出张量 `output_tensor`。

Softmax激活函数的作用是将输入张量的每个元素进行指数运算，然后对每个样本的元素进行归一化，使得每个样本的元素和为1。在示例中，输入张量中的每个元素都经过了Softmax函数的计算，得到了相应的输出值。

`Softmax()`函数的可选参数`dim`用于指定Softmax操作的维度。默认情况下，它的值为`None`，表示在最后一个维度上进行Softmax操作。在示例中，我们通过`dim=1`指定在输入张量的第二个维度上进行Softmax操作，即对每一行进行Softmax操作。

Softmax激活函数常用于多分类问题，可以将输出值解释为样本属于每个类别的概率分布。它可以用于模型的最后一层，以便对模型的输出进行归一化处理。


```python
import torch.nn.functional as F

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

class LogisticRegressionModel(torch.nn.Module):
	def __init__(self):
		super(LogisticRegressionModel, self).__init__() 
		self.linear = torch.nn.Linear(1, 1)
	def forward(self, x):
		y_pred = F.sigmoid(self.linear(x))
		return y_pred

model = LogisticRegressionModel() 
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  

for epoch in range(1000):
	y_pred = model(x_data)
	loss = criterion(y_pred, y_data)
	print(epoch, loss.item())
	optimizer.zero_grad() 
	loss.backward()
	optimizer.step()
```