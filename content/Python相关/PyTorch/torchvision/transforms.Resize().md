在PyTorch的`torchvision`库中，`transforms.Resize`函数用于调整图像的大小。这个函数通常用于数据预处理，特别是在训练图像分类、目标检测或分割等任务时。

[transforms.Resize()官网](https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=resize#torchvision.transforms.Resize)

```python
from torchvision import transforms
transforms.Resize(size, interpolation=None)
```

### 参数介绍
- `size`：一个**新的尺寸**，可以是单个值，表示宽度或高度，或者是一个元组，表示宽度和高度。如果`size`是一个元组，那么`interpolation`参数必须是None，因为不同的尺寸需要不同的插值方法。

- `interpolation`：可选参数，指定插值方法。它可以是以下值：
  - `None`（默认）：不进行插值，直接缩放。
  - `PIL.Image.NEAREST`：最近邻插值。
  - `PIL.Image.BILINEAR`：双线性插值。
  - `PIL.Image.BICUBIC`：双三次插值。
  - `PIL.Image.LANCZOS`：拉普拉斯插值。

### 应用举例
下面是一个使用`transforms.Resize`的简单例子：
```python
import torch
import torchvision
import torchvision.transforms as transforms
# 假设我们有一些输入图像
image = torch.randn(3, 100, 100)  # 创建一个随机图像
# 使用Resize函数将图像大小调整为新的尺寸
new_size = (200, 200)
resized_image = transforms.Resize(new_size)(image)

# 输出调整后的图像大小
print(resized_image.shape)  # 输出: torch.Size([3, 200, 200])
```
在这个例子中，我们首先创建了一个随机图像，然后使用`transforms.Resize`函数将其大小调整为`200x200`。`resized_image`张量现在具有新的尺寸。
