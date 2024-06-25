在PyTorch中，`transforms.ToTensor()`函数用于将PIL图像或NumPy数组**转换为张量**（Tensor）格式。
**函数定义**：
```python
class ToTensor(object):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): 输入的图像。

        Returns:
            Tensor: 转换为张量的图像。
        """
        # 转换为张量
        return F.to_tensor(pic)
```
**参数**：
该函数没有独立的参数，它是一个类，其实例可以直接调用。
**示例**：
以下是使用`transforms.ToTensor()`函数将PIL图像或NumPy数组转换为张量的示例：
```python
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 示例1：将PIL图像转换为张量
# 读取PIL图像
image_pil = Image.open('image.jpg')

# 创建ToTensor转换实例
to_tensor = transforms.ToTensor()

# 将PIL图像转换为张量
image_tensor = to_tensor(image_pil)

# 打印张量形状和数据类型
print(image_tensor.shape)
print(image_tensor.dtype)

# 示例2：将NumPy数组转换为张量
# 创建NumPy数组
image_np = np.array(image_pil)

# 将NumPy数组转换为张量
image_tensor = to_tensor(image_np)

# 打印张量形状和数据类型
print(image_tensor.shape)
print(image_tensor.dtype)
```

在示例1中，我们首先使用PIL库的`Image.open()`函数读取一张图像，然后创建了`transforms.ToTensor()`的实例`to_tensor`。接下来，我们调用`to_tensor`实例并传入PIL图像，将其转换为张量格式。最后，我们打印出张量的形状和数据类型。

在示例2中，我们创建了一个NumPy数组`image_np`，它是通过将PIL图像转换为NumPy数组得到的。然后，我们使用`to_tensor`实例将NumPy数组转换为张量。最后，我们打印出张量的形状和数据类型。

通过使用`transforms.ToTensor()`函数，我们可以方便地将PIL图像或NumPy数组转换为PyTorch中的张量格式，以便进行深度学习模型的输入和处理。