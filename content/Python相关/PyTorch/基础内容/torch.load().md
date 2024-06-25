在Python的PyTorch库中，`torch.load()`函数用于加载之前使用`torch.save()`函数保存的PyTorch模型或张量。
**函数定义**：
```python
torch.load(f, map_location=None, pickle_module=<module 'pickle' from '...'>)
```

**参数**：
以下是`torch.load()`函数中常用的参数：

- `f`：要加载的文件路径（包括文件名），可以是字符串或类文件对象。

- `map_location`（可选）：用于将加载的对象映射到特定设备的函数或字典。如果不指定，则加载到与原始设备相同的设备上。

- `pickle_module`（可选）：用于序列化和反序列化对象的模块。默认值为Python内置的`pickle`模块。

**示例**：
以下是使用`torch.load()`函数加载保存的PyTorch模型和张量的示例：

```python
import torch

# 加载保存的模型
model = torch.load("model.pt")

# 加载保存的张量
tensor = torch.load("tensor.pt")
```

在上述示例中，我们首先导入了`torch`库。

然后，我们使用`torch.load()`函数加载之前保存的模型文件`model.pt`。加载后，模型会被赋值给变量`model`。

接下来，我们使用`torch.load()`函数加载之前保存的张量文件`tensor.pt`。加载后，张量会被赋值给变量`tensor`。

通过运行上述代码，我们可以成功加载之前保存的模型和张量，并在后续的代码中使用它们。

除了上述示例中的参数，`torch.load()`函数还有其他可用的参数和选项，用于更精细的控制加载的方式和行为。例如，可以使用`map_location`参数将加载的对象映射到指定的设备上，或者使用`pickle_module`参数指定用于反序列化的模块。详细的参数说明可以参考PyTorch官方文档。