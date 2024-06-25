在PyTorch中，`torch.utils.data.TensorDataset()`函数用于**创建一个张量数据集**（Tensor Dataset）。张量数据集是一个包含多个张量的数据集，每个张量代表一个样本的**特征和标签**。`TensorDataset`将这些张量组合成对应的**样本**。

将x和y训练数据存入到dataset中。TensorDataset:把输入的两类数据进行一一对应？

```python
torch.utils.data.TensorDataset(*tensors)
```

**参数**：
- `*tensors`：一个或多个张量参数，**每个张量代表一个样本的特征和标签**。所有张量的第一个维度（通常是样本数量）必须相同。例如，**(features, labels)** 创建一个张量数据集，其中 `features` 和 `labels` 是两个张量。

**示例**：
```python
import torch
from torch.utils.data import TensorDataset

# 创建特征张量
features = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

# 创建标签张量
labels = torch.tensor([0, 1, 1])

# 创建张量数据集
dataset = TensorDataset(features, labels)

# 遍历数据集
for sample in dataset:
    print(sample)
```

**输出示例**：
```
(tensor([1, 2, 3]), tensor(0))
(tensor([4, 5, 6]), tensor(1))
(tensor([7, 8, 9]), tensor(1))
```

在上述示例中，我们首先创建了两个张量，`features` 和 `labels`，分别表示样本的特征和标签。`features` 是一个形状为 `(3, 3)` 的特征张量，每一行代表一个样本的特征。`labels` 是一个形状为 `(3,)` 的标签张量，每个元素代表一个样本的标签。

然后，我们使用 `TensorDataset` 创建了一个张量数据集 `dataset`，将特征张量和标签张量作为参数传递给函数。

最后，我们使用一个 `for` 循环遍历数据集 `dataset`，并打印每个样本。可以看到，每个样本由一个特征张量和一个标签张量组成，它们分别表示样本的特征和标签。

`torch.utils.data.TensorDataset()`函数在进行数据处理和模型训练时非常有用，它能够方便地将特征和标签组合成对应的样本，并作为输入提供给数据加载器或模型。你可以使用多个张量创建一个张量数据集，以适应不同的数据集结构和任务需求。


### 疑问1
为什么不能直接使用d2l.synthetic_data()生成的张量训练集？其实d2l知识正对动手学深度学习里的内容