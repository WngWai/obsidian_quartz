在 PyTorch 中，`torch.mul()` 函数用于对两个张量进行逐元素相乘（element-wise multiplication）。它执行逐元素相乘的操作，即将两个张量中对应位置的元素相乘。

以下是 `torch.mul()` 函数的基本信息：

**所属包：** torch

**定义：**
```python
torch.mul(input, other, out=None)
```

**参数介绍：**
- `input`：第一个输入张量。
- `other`：第二个输入张量。
- `out`：可选，用于存储结果的输出张量。

**功能：**
对两个张量进行逐元素相乘。

**举例：**
```python
import torch

# 创建两个张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# 逐元素相乘
result = torch.mul(tensor1, tensor2)

# 打印结果
print(result)
```

**输出：**
```
tensor([ 4, 10, 18])
```

在上述示例中，`torch.mul()` 被用于对两个张量 `tensor1` 和 `tensor2` 进行逐元素相乘。结果是一个新的张量 `[4, 10, 18]`，其中每个元素都是对应位置上两个张量的元素相乘的结果。