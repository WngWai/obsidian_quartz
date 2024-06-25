`torchvision`是PyTorch中的一个包，提供了**用于计算机视觉任务的工具和函数**。
torchvision是一个用于计算机视觉任务的库，提供了载入和处理图像数据集、定义图像转换、构建神经网络模型等功能。它包含了常见的计算机视觉数据集、预训练的模型以及图像操作的工具函数。

### 数据集和数据加载：
[[torchvision.datasets]]提供了常**用的计算机视觉数据集**，如MNIST、CIFAR-10、CIFAR-100、ImageNet等。
[[torchvision.datasets.ImageFolder()]]数据集
[[torchvision.datasets.FashionMNIST()]]数据集

### 图像变换和增强：
torchvision.transforms在torch2.0中被`torchvision.transforms.v2`模块给替代了。提供了**数据加载器**（DataLoader）和**数据转换器**（DataTransformer），用于加载和预处理图像数据。提供了多种图像预处理的方法，这些方法可以被用来对图像进行缩放、裁剪、归一化等操作，以便于模型的训练和测试。
```python
？？？
from torch.torchvision.transforms import v2 as transforms
```

   - `torchvision.transforms`: 提供了多种图像变换和增强操作，如随机裁剪、缩放、翻转、旋转、归一化等，用于数据增强和预处理。
   
   
   
   - `torchvision.transforms.Compose`: 将多个图像变换操作组合在一起，形成一个变换流水线。

  
[[transforms.Resize()]] 调整图像大小


- **基本操作**：如`Resize`、`CenterCrop`、`Normalize`等。
- **数据增强**：如`RandomHorizontalFlip`、`RandomCrop`等。

？？？
- `Resize`, `CenterCrop`, `RandomCrop` 等用于调整图像尺寸和形状
- `ToTensor`, `Normalize` 等用于数据类型转换和标准化
- 数据增强相关的 `RandomHorizontalFlip`, `RandomRotation` 等


### 模型
模型预训练：
   - `torchvision.models`: 提供了在大规模图像数据上预训练的深度学习模型，如AlexNet、VGG、ResNet、Inception等。
   - `torchvision.models.resnet18(pretrained=True)`: 加载预训练的ResNet-18模型。

`torchvision.models`提供了一系列预训练的模型，可以直接用于推理或者作为预训练模型进行进一步的训练。这些模型包括但不限于：

- **分类(Classification)**：如ResNet、VGG、AlexNet等。
- **目标检测(Object Detection)**：如Faster R-CNN、SSD等。
- **分割(Segmentation)**：如FCN、DeepLab等。
- **人体关键点检测(Pose Estimation)**：如Keypoint R-CNN。

### 工具（Utils）
模型评估和可视化：
   - `torchvision.utils.make_grid`: 将模型输出的图像拼接成一个网格图像。
   - `torchvision.utils.draw_bounding_boxes`: 在图像上绘制边界框。


#### 图像工具：
   - `torchvision.utils.save_image`: 保存图像到文件。
   - `torchvision.utils.make_grid`: 将多个图像拼接成一个网格图像。

### 其他

`torchvision` 是 PyTorch 的一个专门用于计算机视觉任务的库，提供了一系列用于加载、预处理和可视化图像数据集的工具，同时也包含了一些经典的计算机视觉模型。以下是 `torchvision` 模块中一些主要的函数，按照功能进行分类：

### 数据加载和预处理：

1. **数据加载：**
    - `torchvision.datasets` 模块包含了多个常见的计算机视觉数据集，如 `CIFAR10`、`MNIST` 等。
    - `torchvision.transforms` 模块提供了各种图像预处理的函数，如调整大小、裁剪、翻转等。

### 模型和预训练模型：

1. **模型定义：**
    - `torchvision.models` 模块包含了一些常见的计算机视觉模型，如 `ResNet`、`VGG`、`AlexNet` 等。

2. **预训练模型：**
    - `torchvision.models` 中的模型类可以通过 `pretrained=True` 参数加载预训练模型。

### 图像可视化：

1. **图像显示：**
    - `torchvision.utils` 模块中的 `make_grid` 函数用于将多张图像拼接成一个网格。

2. **图像保存：**
    - `torchvision.utils` 模块中的 `save_image` 函数用于将图像保存到文件。

### 其他功能：

1. **计算图像统计信息：**
    - `torchvision.utils` 模块中的 `calculate_mean_and_std` 函数用于计算图像数据集的均值和标准差。

2. **获取图像文件路径列表：**
    - `torchvision.datasets.ImageFolder` 类可以用于从包含图像的文件夹中获取图像路径列表。

3. **图像变换：**
    - `torchvision.transforms` 模块中的各种变换函数用于对图像进行不同的变换操作，如随机旋转、亮度调整等。

4. **图像数据加载器：**
    - `torchvision.transforms` 模块中的 `DataLoader` 类用于加载图像数据，并可以结合 `Dataset` 使用。

这些函数和类使得在计算机视觉任务中能够更方便地处理和操作图像数据，同时能够利用预训练模型进行迁移学习。在使用 `torchvision` 时，通常结合 `torch.utils.data` 模块一起使用，以构建用于训练和评估的数据加载器。