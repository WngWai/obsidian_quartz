在 PyTorch 中，`sum()` 是一个 Tensor 对象的方法，用于计算 Tensor 中元素的总和。它可以对整个 Tensor 进行求和，也可以**沿着指定的维度进行求和**。
```python
sum(dim=None, keepdim=False, dtype=None)
```

- `dim`：可选参数，用于指定沿着**哪个维度**进行求和。默认情况下，对**所有维度**进行求和，返回一个标量值。
`指定哪个维度，哪个维度“拍扁”为0而非1，压缩为1需要设置keepdim。对于2*3矩阵来说。shape中的第0号元素（行）从2变成了0，沿着行的方向进行求和，形成一个含有三个元素的向量。`

![[Pasted image 20231008215057.png]]

- `keepdim`：可选参数，用于指定是否保持输出 Tensor 的**维度和输入Tensor 相同**。默认为**False**，即输出 Tensor 的维度会缩减。
`二维矩阵的理解，虽然压缩为一维向量，但形式上仍保留二维矩阵！`

关于axis=0，1，2的差异
![[Pasted image 20230925212818.png]]

- `dtype`：可选参数，用于指定输出Tensor的数据类型。如果未指定，则使用输入Tensor 的数据类型。

1. 对整个 Tensor 进行求和：
   ```python
   import torch

   x = torch.tensor([1, 2, 3, 4, 5])
   total_sum = x.sum()
   print(total_sum)
   # 输出: tensor(15)
   ```
在这个例子中，`sum()` 方法对整个 Tensor 进行求和，返回的是一个标量值。

2. 沿着指定的维度进行求和：
   ```python
   import torch

   x = torch.tensor([[1, 2, 3], [4, 5, 6]])
   row_sum = x.sum(dim=1)
   print(row_sum)
   # 输出: tensor([ 6, 15])
   ```
在这个例子中，`sum(dim=1)` 对输入的二维 Tensor **沿着第一个维度（行）** 进行求和，返回的是一个一维 Tensor，包含每行元素的总和。

3. 保持输出 Tensor 的维度与输入 Tensor 相同：
   ```python
   import torch

   x = torch.tensor([[1, 2, 3], [4, 5, 6]])
   column_sum = x.sum(dim=0, keepdim=True)
   print(column_sum)
   # 输出: tensor([[5, 7, 9]])
   ```
在这个例子中，`sum(dim=0, keepdim=True)` 对输入的二维 Tensor 沿着**第二个维度（列）** 进行求和，并保持输出 Tensor 的维度与输入 Tensor 相同。

`sum()` 方法返回一个 Tensor 对象，其中包含了求和结果。根据参数的设置，可以对整个 Tensor 进行求和或沿着指定的维度进行求和。


### 关于sum进行反向求导的理解
？？？
对模型训练代码中l.sum().backward()的粗浅理解 - 吉平.集的文章 - 知乎
https://zhuanlan.zhihu.com/p/604681583