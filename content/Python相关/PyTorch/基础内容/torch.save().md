在Python的PyTorch库中，`torch.save()`函数用于将PyTorch模型或张量保存到磁盘上。

**函数定义**：
```python
torch.save(obj, f, pickle_module=<module 'pickle' from '...'>, pickle_protocol=2, _use_new_zipfile_serialization=True)
```

**参数**：
以下是`torch.save()`函数中常用的参数：

- `obj`：要保存的对象，可以是PyTorch模型、张量或字典等。

- `f`：保存的文件路径（包括文件名），可以是字符串或类文件对象。

- `pickle_module`（可选）：用于序列化和反序列化对象的模块。默认值为Python内置的`pickle`模块。

- `pickle_protocol`（可选）：用于指定pickle协议的版本。默认值为2，即Python 2.x和3.x都兼容的协议。

- `_use_new_zipfile_serialization`（可选）：用于指定是否使用新的Zip文件序列化机制。默认值为`True`。

**示例**：
以下是使用`torch.save()`函数保存PyTorch模型和张量的示例：

```python
import torch

# 创建一个PyTorch模型
model = torch.nn.Linear(10, 2)

# 保存模型到文件
torch.save(model, "model.pt")

# 创建一个张量
tensor = torch.tensor([1, 2, 3, 4, 5])

# 保存张量到文件
torch.save(tensor, "tensor.pt")
```

在上述示例中，我们首先导入了`torch`库。

然后，我们创建了一个简单的PyTorch模型`model`，它是一个线性层。

接下来，我们使用`torch.save()`函数将模型保存到名为`model.pt`的文件中。

然后，我们创建了一个张量`tensor`，它包含了一些简单的数值。

最后，我们使用`torch.save()`函数将张量保存到名为`tensor.pt`的文件中。

通过运行上述代码，模型和张量会被保存到指定的文件中，可以通过后续的`torch.load()`函数来加载它们，并在其他地方使用。

除了上述示例中的参数，`torch.save()`函数还有其他可用的参数和选项，用于更精细的控制保存的方式和格式。详细的参数说明可以参考PyTorch官方文档。