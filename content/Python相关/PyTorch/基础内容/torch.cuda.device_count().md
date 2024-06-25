在 PyTorch 中，`torch.cuda.device_count()` 函数用于获取当前系统中可用的 GPU 数量。以下是该函数的基本信息：

**返回当前系统中可用的 GPU 数量**

### 定义：
```python
torch.cuda.device_count()
```

### 参数介绍：
该函数没有额外的参数。

### 举例：
```python
import torch

# 获取可用的 GPU 数量
gpu_count = torch.cuda.device_count()

print("Number of available GPUs:", gpu_count)
```

### 注意事项：
- 如果系统中**没有安装 GPU 或者 GPU 不可用**，`torch.cuda.device_count()` 将返回 0。

这个函数对于在训练深度学习模型时检查系统中 GPU 的可用性是很有用的。在使用 PyTorch 进行 GPU 加速计算时，可以使用此函数来确定系统上可以利用的 GPU 数量。