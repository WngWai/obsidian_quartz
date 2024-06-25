t.to(device)
net.to(device)可以真个模型都移到GPU上去！

在PyTorch中，`t.to()`函数用于**将张量转换到指定的设备**（如CPU或GPU）或指定的数据类型。

指定计算使用CPU还是GPU，存在GPU和CPU数据不相通的情况。以及读取数据过大，导致超过内存的情况

**方法定义**：
```python
t.to(device=None, dtype=None, non_blocking=False, copy=False)
```
**参数**：
- `device`（设备，可选）：指定目标设备，可以是`torch.device`对象、字符串（如`'cuda'`）或`int`类型的设备索引。如果未指定，则默认为当前设备。

'cpu'，CPU

'cuda'，GPU，一般指NVIDA,CUDA？AMD的怎么办？

- `dtype`（数据类型，可选）：指定**目标数据类型**。可以是`torch.dtype`对象或与PyTorch兼容的字符串（如`'torch.float32'`）。如果未指定，则默认为原始张量的数据类型。

- `non_blocking`（布尔值，可选）：如果为`True`，则会尝试以异步方式将张量转移到目标设备，否则以同步方式转移。默认为`False`。

- `copy`（布尔值，可选）：如果为`True`，则会创建原始张量的副本，并将副本转移到目标设备。如果为`False`，则只会返回原始张量的视图，并将其转移到目标设备。默认为`False`。

**返回值**：
返回转换后的张量。

**示例**：
以下是使用`to()`方法的示例：

```python
import torch

# 创建一个张量
t = torch.tensor([1, 2, 3])

# 将张量转移到GPU
t_gpu = t.to('cuda')
print(t_gpu)
# 输出: tensor([1, 2, 3], device='cuda:0')

# 将张量转移到CPU
t_cpu = t_gpu.to('cpu')
print(t_cpu)
# 输出: tensor([1, 2, 3])

# 将张量转换为双精度浮点型
t_double = t.to(dtype=torch.float64)
print(t_double)
# 输出: tensor([1., 2., 3.], dtype=torch.float64)
```

在上述示例中，我们首先创建了一个张量 `t`，包含元素 `[1, 2, 3]`。

然后，我们使用`to()`方法将该张量转移到GPU，通过将字符串 `'cuda'` 作为参数传递给`to()`方法，我们将张量 `t` 转移到当前可用的GPU设备。结果是一个新的张量 `t_gpu`，其存储在GPU上，并显示设备信息。

接下来，我们使用`to()`方法将 `t_gpu` 张量转移到CPU，通过将字符串 `'cpu'` 作为参数传递给`to()`方法，我们将张量 `t_gpu` 转移到CPU。结果是一个新的张量 `t_cpu`，其存储在CPU上。

最后，我们使用`to()`方法将 `t` 张量转换为双精度浮点型。通过将 `torch.float64` 作为参数传递给`to()`方法，我们将张量 `t` 的数据类型转换为双精度浮点型。结果是一个新的张量 `t_double`，其数据类型为 `torch.float64`。

`to()`方法对于在不同设备间传输张量或更改张量的数据类型非常有用，可以灵活地管理张量的位置和数据类型。


### 实现GPU加速的两种设置

```python
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 指明调用的GPU为0,1号

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 指明调用的GPU为1号
```
