在PyTorch中，`torchvision.datasets.FashionMNIST()`函数用于**加载FashionMNIST数据集**，该数据集包含了**10种不同类型的时尚商品图像**。

**函数定义**：
```python
torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
```

**参数**：
- `root`（必需）：指定**FashionMNIST数据集**的**根目录**。如果在root中没有找到文件，从网上下载到root。如"./data/"
datasets.FashionMNIST就是一个Datasets子类，data是这个类的一个实例？？？


- `train`（可选）：布尔值，指示**是否加载训练集**。默认为`True`，加载训练集。如果为`False`，加载测试集。

- `transform`（可选）：指定**数据集的转换操作**。默认为`None`，表示不进行任何转换。**trans**将图片转换为张量。
trans = transforms.ToTensor()
transform=trans

- `target_transform`（可选）：指定**目标（标签）的转换操作**。默认为`None`，表示不进行任何转换。

- `download`（可选）：布尔值，指示**是否下载FashionMNIST数据集**。默认为`False`，表示不下载。如果**数据集未下载，则会自动下载**，保存的位置是就是root的位置。TRUE进行**联网下载**。

**示例**：
以下是使用`torchvision.datasets.FashionMNIST()`函数加载FashionMNIST数据集的示例：
```python
import torch
from torchvision import datasets, transforms

# 定义数据集的转换操作
transform = transforms.ToTensor()

# 加载训练集
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

# 加载测试集
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 获取数据集的大小
train_size = len(train_dataset)
test_size = len(test_dataset)

# 打印数据集的大小
print("Train dataset size:", train_size)
print("Test dataset size:", test_size)

# 获取单个样本
image, label = train_dataset[0]

# 打印样本的形状和标签
print("Image shape:", image.shape)
print("Label:", label)
```

在上述示例中，我们首先定义了数据集的转换操作`transform`，这里使用了`transforms.ToTensor()`函数将图像转换为张量格式。接下来，我们使用`datasets.FashionMNIST()`函数加载训练集和测试集，指定数据集的根目录、转换操作和下载选项。然后，我们分别获取训练集和测试集的大小，并打印出来。

最后，我们使用索引访问训练集中的第一个样本，获取图像和标签。我们打印出图像的形状和标签，以便查看数据集的内容。

通过使用`torchvision.datasets.FashionMNIST()`函数，我们可以方便地加载FashionMNIST数据集，并进行进一步的数据预处理和模型训练。