在 PyTorch 中，向量的叉乘通常使用 `torch.cross()` 函数进行表示。叉乘（cross product）是**两个向量的一种二元运算，产生一个与这两个向量都垂直的新向量**。在 3D 空间中，叉乘的结果是一个新的向量，它垂直于原始向量构成的平面。

以下是 `torch.cross()` 函数的基本信息：

**所属包：** torch

**定义：**
```python
torch.cross(input, other, dim=-1, out=None)
```

**参数介绍：**
- `input`：第一个输入向量。
- `other`：第二个输入向量。
- `dim`：表示在哪个维度上执行叉乘操作。默认为 -1，表示最后一个维度。
- `out`：可选，用于存储结果的输出张量。

**功能：**
计算两个向量的叉乘。

**举例：**
```python
import torch

# 创建两个3维向量
vector1 = torch.tensor([1, 2, 3])
vector2 = torch.tensor([4, 5, 6])

# 计算叉乘
cross_product = torch.cross(vector1, vector2)

# 打印结果
print(cross_product)
```

**输出：**
```
tensor([-3,  6, -3])
```

在上述示例中，`torch.cross()` 被用于计算两个3维向量 `vector1` 和 `vector2` 的叉乘。得到的结果是一个新的3维向量 `[-3, 6, -3]`，该向量垂直于 `vector1` 和 `vector2` 所在的平面。