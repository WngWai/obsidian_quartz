在PyTorch中，`torch.utils.data.DataLoader()`函数用于创建一个**数据加载器**（Data Loader）。数据加载器用于**从数据集中加载小批量数据并进行迭代，以便于模型的训练和评估**。`DataLoader`提供了一些参数来配置数据加载的方式和行为。

dataloader实例对象

本身就是可迭代对象，for batch in dataloader。

```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,  pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

![[Pasted image 20231025192246.png]]
根据batch_size加载dataset，得到小批量样本。

**参数**：
- `dataset`：数据集对象

- batch_size可选参数，用于指定**每个小批量的样本数量**。默认值为**1**，表示每次加载一个样本。

- shuffle进行随机抽样，可选参数，用于指定**是否对数据进行洗牌**（**随机排序**）。默认值为**False**，表示不进行洗牌。
一般training集要随机抽样，testing集不用随机抽样。

- `sampler`：可选参数，用于**指定自定义的采样器**。如果指定了`sampler`，则忽略 `shuffle` 参数。

- `batch_sampler`：可选参数，用于指定自定义的小批量采样器。如果指定了 `batch_sampler`，则忽略 `batch_size`、`shuffle` 和 `sampler` 参数。

- `num_workers`：**进程数**，可选参数，用于指**定数据加载的并行工作线程数量**。默认值为**0**，表示在主进程中进行数据加载。Linux系统一版是4或8
不是越大越好，毕竟还要进程干别的事情！

- `collate_fn`：可选参数，用于指定自定义的样本组合函数。默认值为None，表示使用默认的组合函数。

- `pin_memory`：可选参数，用于指定是否将加载的数据存储在固定的内存位置中。默认值为False。

- `drop_last`：可选参数，用于指定当数据集大小不是批量大小的整数倍时，是否**丢弃最后一个不完整的小批量**。默认值为False，不丢弃。

- `timeout`：可选参数，用于指定数据加载超时时间（以秒为单位）。默认值为0，表示没有超时限制。

- `worker_init_fn`：可选参数，用于指定每个工作线程的初始化函数。默认值为None。

- `multiprocessing_context`：可选参数，用于指定多进程加载数据的上下文。默认值为None。
**示例**：
```python
import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils import data

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 创建数据集对象
dataset = CustomDataset([1, 2, 3, 4, 5])

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

# 遍历数据加载器
for batch in dataloader:
    print(batch)
```

**输出示例**：
```python
tensor([1, 2])
tensor([5, 3])
tensor([4])
```

在上述示例中，我们首先定义了一个自定义的数据集类 `CustomDataset`，该类继承自 `torch.utils.data.Dataset`。我们在 `CustomDataset` 中实现了 `__len__` 和 `__getitem__` 方法，以便能够被数据加载器加载。
然后，我们创建了一个数据集对象 `dataset`，将自定义数据 `[1, 2, 3, 4, 5]` 传递给数据集对象。
接下来，我们使用 `DataLoader` 创建了一个数据加载器 `dataloader`。我们指定了 `batch_size=2`，表示每次加载两个样本；`shuffle=True`，表示对数据进行洗牌；`num_workers=2`，表示使用两个工作线程进行数据加载。
最后，我们使用一个 `for` 循环遍历数据加载器 `dataloader`，并打印每个小批量的数据。可以看到，数据加载器每次返回一个小批量的张量，其中包含两个样本。由于指定了 `shuffle=True`，每个小批量的样本顺序是随机的。
