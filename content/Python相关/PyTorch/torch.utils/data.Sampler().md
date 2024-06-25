在 PyTorch 的 `torch.utils.data` 模块中，`Sampler` 类是一个抽象基类，用于定义自定义数据集的采样器。由于 `Sampler` 是一个抽象基类，实际使用时需要使用其子类，如 `SequentialSampler`、`RandomSampler` 等。以下是参数的详细介绍和示例：
采样器类，用于定义数据集的采样策略，如随机采样、有序采样等。

参数：
- `data_source`：数据集对象，通常是 `torch.utils.data.Dataset` 的子类对象。该参数是在子类的构造函数中传递的。
示例：
```python
import torch
from torch.utils.data import Dataset, Sampler

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

# 自定义采样器类
class MySampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self):
        # 返回自定义的迭代器
        return iter(range(len(self.data_source)))
    
    def __len__(self):
        return len(self.data_source)

# 创建自定义数据集对象
dataset = MyDataset([1, 2, 3, 4, 5])

# 创建自定义采样器对象
sampler = MySampler(dataset)

# 使用采样器进行数据加载
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)

# 遍历数据加载器
for batch in dataloader:
    print(batch)
```

在上述示例中，我们首先定义了一个自定义数据集类 `MyDataset`，其中包含了数据的获取方式和数据集的长度等方法。然后，我们定义了一个自定义采样器类 `MySampler`，继承自 `Sampler` 类，并实现了 `__iter__()` 和 `__len__()` 方法。`__iter__()` 方法返回一个自定义的迭代器对象，用于确定数据样本的顺序。最后，我们使用自定义采样器对象 `sampler` 在数据加载器中进行数据加载。

输出结果为：
```
tensor([1, 2])
tensor([3, 4])
tensor([5])
```

这表示我们成功地通过自定义采样器 `MySampler` 对数据集进行了自定义的顺序采样。在遍历数据加载器时，每个批次的数据按照自定义采样器中确定的顺序进行加载。

请注意，以上示例中的 `MySampler` 类只是一个简单的示例，它按顺序返回数据样本的索引。您可以根据需求自定义更复杂的采样器，例如随机采样器、带权重的采样器等，以满足不同的数据采样需求。