在Python的`torch.utils`包中，`random_split()`函数用于将一个数据集随机划分为训练集和验证集或测试集。它是`torch.utils.data.Dataset`类的一个方法。`random_split()`函数允许你指定划分的大小或比例，以及是否设置随机种子。
```python
random_split(dataset, lengths, generator=None)
```
参数说明：
- `dataset`: 必需，表示原始数据集对象，通常是`torch.utils.data.Dataset`的一个实例。
- `lengths`: 必需，表示划分的大小或比例。可以是整数列表，表示每个划分的大小，或浮点数列表，表示每个划分的比例。列表中的值应该总和为原始数据集的长度。
- `generator`（可选）: 表示随机种子生成器的对象。默认值为`None`，表示使用默认的随机种子生成器。
示例：
以下示例演示了如何使用`random_split()`函数将数据集随机划分为训练集和验证集：

首先，确保已安装`torch`库。你可以使用以下命令进行安装：
```python
pip install torch
```
接下来，导入相关库：
```python
import torch
from torch.utils.data import Dataset, random_split
```

然后，我们创建一个自定义的数据集类`CustomDataset`：
```python
class CustomDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
```

现在，我们可以创建原始数据集对象并进行随机划分：
```python
# 创建原始数据集对象
dataset = CustomDataset()

# 指定划分的大小或比例
lengths = [3, 2]

# 随机划分数据集
train_dataset, val_dataset = random_split(dataset, lengths)

# 打印划分后的数据集长度
print(len(train_dataset))
print(len(val_dataset))
```

输出：
```
3
2
```

这个示例展示了如何使用`random_split()`函数将数据集随机划分为训练集和验证集。我们首先定义了一个自定义的数据集类`CustomDataset`，然后创建了一个原始数据集对象`dataset`。接下来，我们指定了划分的大小或比例，其中`lengths`列表表示划分为3个训练样本和2个验证样本。最后，我们使用`random_split()`函数将数据集随机划分为训练集和验证集，并分别将它们存储在`train_dataset`和`val_dataset`中。我们可以通过`len(train_dataset)`和`len(val_dataset)`获取划分后的训练集和验证集的长度。

请注意，`random_split()`函数返回的是划分后的数据集对象，因此可以像处理任何其他数据集对象一样使用它们。