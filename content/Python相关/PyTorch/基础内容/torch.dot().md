在 PyTorch 中，`torch.dot()` 函数用于计算两个张量的**点积（内积）**。点积是两个向量**对应元素相乘后的和**，通常用于计算两个向量的相似性或投影。

以下是 `torch.dot()` 函数的基本信息：

**所属包：** torch

**定义：**
```python
torch.dot(tensor1, tensor2)
```

**参数介绍：**
- `tensor1`：第一个输入张量。
- `tensor2`：第二个输入张量。

**功能：**
计算两个张量的点积。

**举例：**
```python
import torch

# 创建两个张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# 计算点积
dot_product = torch.dot(tensor1, tensor2)

# 打印结果
print(dot_product)
```

**输出：**
```
tensor(32)
```

在上述示例中，`torch.dot()` 被用于计算两个张量 `tensor1` 和 `tensor2` 的点积。具体计算是将它们对应位置的元素相乘，然后将结果相加，得到点积的值。在这个例子中，点积为 1\*4 + 2\*5 + 3\*6 = 32。
