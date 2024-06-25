在PyTorch中，`torch.cuda.is_available()`函数用于检查**当前系统是否支持CUDA加速**，并且是否有可用的CUDA设备，返回**布尔值**。

**函数定义**：
```python
torch.cuda.is_available()
```

**返回值**：
返回一个布尔值，表示CUDA是否可用。如果CUDA可用且至少有一个CUDA设备可用，则返回`True`，否则返回`False`。

**示例**：
以下是使用`torch.cuda.is_available()`函数的示例：

```python
import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
```

在上述示例中，我们使用`torch.cuda.is_available()`函数来检查CUDA是否可用。如果CUDA可用，则打印出"CUDA is available"；否则打印出"CUDA is not available"。

该函数对于在使用PyTorch进行GPU加速计算之前进行必要的检查非常有用。如果CUDA可用且至少有一个CUDA设备可用，我们可以利用CUDA设备进行加速计算。否则，我们可以选择在CPU上执行计算。