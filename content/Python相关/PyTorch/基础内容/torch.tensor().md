在PyTorch中，`torch.tensor()`函数是用于**创建张量**（Tensor）的主要方式之一，根据提供的数据创建一个新的张量。它可以从Python列表、NumPy数组或其他可迭代对象中创建张量，并可选择指定数据类型和设备。
```python
torch.tensor(data, dtype=None, device=None, requires_grad=False)
```

- `data`：用于创建张量的数据。可以是Python列表、NumPy数组、Python标量或其他可迭代对象。

- `dtype`（可选）：指定创建张量的数据类型。默认为None，表示自动推断数据类型。
如torch.float32，torch.float64(比32多一个小数点？)

- `device`（可选）：指定**张量所在的设备**。默认为None，表示使用默认设备（通常是CPU）。
device="cpu"
device="cuda:0"

- `requires_grad`（可选）：是否需要求导，从而系统会自动调用autograd记录操作，**跟踪张量的梯度信息**。默认为False，表示不跟踪梯度。
`理解为张量不仅是数据的载体，也带有微分运算的属性。`
要想使得Tensor使用autograd功能，只需要设置 tensor.requries grad=True，自动求导

**示例**：
```python
import torch
import numpy as np

# 从Python列表创建张量
data_list = [1, 2, 3, 4, 5]
t1 = torch.tensor(data_list)
print("从Python列表创建的张量：", t1)

# 从NumPy数组创建张量
data_np = np.array([1, 2, 3, 4, 5])
t2 = torch.tensor(data_np)
print("从NumPy数组创建的张量：", t2)

# 指定数据类型和设备
data_float = [1.0, 2.0, 3.0, 4.0, 5.0]
t3 = torch.tensor(data_float, dtype=torch.float32, device='cuda')
print("指定数据类型和设备的张量：", t3)
```

**输出**：
```
从Python列表创建的张量： tensor([1, 2, 3, 4, 5])
从NumPy数组创建的张量： tensor([1, 2, 3, 4, 5])
指定数据类型和设备的张量： tensor([1., 2., 3., 4., 5.], device='cuda:0')
```

在上述示例中，我们使用`torch.tensor()`函数创建了几个不同的张量。首先，我们从Python列表 `data_list` 和NumPy数组 `data_np` 创建了两个张量 `t1` 和 `t2`。默认情况下，这些张量的数据类型和设备将根据提供的数据进行推断。

接下来，我们创建了另一个张量 `t3`，从Python列表 `data_float` 创建。我们通过指定`dtype=torch.float32`将其数据类型设置为浮点型，通过`device='cuda'`将其放置在CUDA设备上（如果可用）。请注意，为了在CUDA上创建张量，你的系统必须具备支持CUDA的GPU。

需要注意的是，`torch.tensor()`函数创建的张量默认情况下不会跟踪梯度（`requires_grad=False`）。如果需要跟踪梯度，请将`requires_grad=True`作为参数传递给函数。

根据提供的数据类型和设备，`torch.tensor()`函数允许你创建具有不同属性的张量，并且非常灵活方便。


### 关于device的详解
张量可以在CPU或GPU上储存，
```windows
nvidia-smi # 查看电脑上的显卡

```
![[Pasted image 20231029164428.png]]
GPU在PyTorch上以cuda:0，cuda:1...依次指定
时间差异：此处可能发现数据转移到GPU后，GPU运算的速度并未提升太多，这是因为x和y太小且运算也较为简单，而且将数据从内存转移到显存还需要花费额外的开销。GPU的优势需在大规模数据和复杂运算下才能体现出来。

### 关于requires_grad的详解
PyTorch会根据计算过程自动生成动态图，然后根据动态图的创建过程进行反向传播，计算得到每个节点的梯度值。
为了能记录张量的梯度，首先需要在创建张量的时候设置requires_grad=True，意味张量加入计算图中，作为计算图的叶子节点参与计算，通过一系列的计算，最后输出结果张量（根节点）。
？？？可微分性？？？
