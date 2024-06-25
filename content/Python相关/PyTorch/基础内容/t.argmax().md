在PyTorch中，`t.argmax()`函数用于返回张量中最大元素的索引。

**函数定义**：
```python
t.argmax(input, dim=None, keepdim=False, out=None)
```

**参数**：
以下是`t.argmax()`函数中常用的参数：

- `input`：输入的张量。

- `dim`：可选参数，指定在哪个维度上进行最大值的搜索。如果未指定，则在整个张量中搜索最大值。

- `keepdim`：可选参数，指定是否保持返回张量的维度与输入张量的维度相同。

- `out`：可选参数，指定输出张量的位置。

**返回值**：
`t.argmax()`函数返回一个新的张量，其中包含输入张量中最大元素的索引。

**示例**：
以下是使用`t.argmax()`函数的示例：

```python
import torch

# 创建一个输入张量
input_tensor = torch.tensor([[1, 2, 3],
                             [4, 5, 6]])

# 在整个张量中搜索最大值的索引
max_index = torch.argmax(input_tensor)
print(max_index)  # 输出: tensor(5)

# 按行搜索最大值的索引
max_index_row = torch.argmax(input_tensor, dim=1)
print(max_index_row)  # 输出: tensor([2, 2])

# 按列搜索最大值的索引
max_index_col = torch.argmax(input_tensor, dim=0)
print(max_index_col)  # 输出: tensor([1, 1, 1])
```

在上述示例中，我们首先创建了一个输入张量`input_tensor`，其中包含了一些数字。

然后，我们使用`torch.argmax()`函数在整个张量中搜索最大值的索引。结果`max_index`是一个标量张量，表示整个张量中最大元素的索引。

接下来，我们使用`dim=1`参数来指定在每一行中搜索最大值的索引。结果`max_index_row`是一个一维张量，表示每一行中最大元素的索引。

最后，我们使用`dim=0`参数来指定在每一列中搜索最大值的索引。结果`max_index_col`是一个一维张量，表示每一列中最大元素的索引。

请注意，上述示例仅演示了基本用法，更多详细的参数和选项可以参考PyTorch的官方文档。