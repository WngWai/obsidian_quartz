在PyTorch中，`t.reshape()`函数用于**改变张量的形状**。它接受一个或多个整数参数作为新的形状，并返回一个具有新形状的张量，而不改变原始张量的数据。下面是参数的详细介绍和举例：

```python
t.reshape()
t.reshape(1,3,28,28)
t.reshape((-1,3,28,28))
```

参数：
- `shape`（**整数或元组**）：指定新的形状。可以传入一个整数表示形状的维度，或者**传入一个元组表示每个维度的大小**。注意，新形状的元素数量应与原始张量的元素**数量保持一致**，否则会引发错误。

(-1,1) ，更一般的(-1,numbers)，-1 表示**根据数组的大小自动计算行数**，元素总量不变，**列多了，行就少了**！

```R
y = [1, 2, 3, 4, 5, 6]

y.reshape((-1, 1))
[[1],
 [2],
 [3],
 [4],
 [5],
 [6]]

y.reshape((-1, 2))
[[1, 2],
 [3, 4],
 [5, 6]]
```


1. 改变张量的维度：
   ```python
   import torch

   x = torch.tensor([[1, 2, 3], [4, 5, 6]])
   print(x.shape)  # 输出: torch.Size([2, 3])

   y = x.reshape((3, 2))
   print(y.shape)  # 输出: torch.Size([3, 2])
   ```
在这个例子中，原始张量`x`的形状是`(2, 3)`，通过`reshape`函数将其改变为`(3, 2)`的形状，得到新的张量`y`。
2. **展平**张量 **-1**：
   ```python
   import torch

   x = torch.tensor([[1, 2, 3], [4, 5, 6]])
   print(x.shape)  # 输出: torch.Size([2, 3])

   y = x.reshape(-1)
   print(y.shape)  # 输出: torch.Size([6])
   ```
在这个例子中，通过将`reshape`函数的参数设置为`-1`，可以将原始张量展平为一个一维张量。
3. 调整张量的维度顺序：
   ```python
   import torch

   x = torch.tensor([[1, 2, 3], [4, 5, 6]])
   print(x.shape)  # 输出: torch.Size([2, 3])

   y = x.reshape((3, 2)).T
   print(y.shape)  # 输出: torch.Size([3, 2])
   ```
在这个例子中，通过`reshape`函数将张量的形状改变为`(3, 2)`，然后使用`.T`操作符将其转置为`(2, 3)`的形状。

注意：`t.reshape()`函数返回的是一个新的张量，如果你想在原地修改张量的形状，可以使用`t.reshape_()`或`t.resize_()`函数。

### 将矩阵拉长的两种方式
`y.reshape((-1, 1))` 和 `y.reshape(-1)` 是对张量 `y` 进行形状重塑的两种不同方式。它们之间的区别在于重塑后的张量的维度。
1. `y.reshape((-1, 1))`：二维张量
将张量 `y` 重塑为一个列向量。
```python
     import torch

     y = torch.tensor([1, 2, 3, 4, 5, 6])
     reshaped_y = y.reshape((-1, 1))

     print(reshaped_y)
     print(reshaped_y.shape)
     
     输出结果：
    
     tensor([[1],
             [2],
             [3],
             [4],
             [5],
             [6]])
     torch.Size([6, 1])
```
在上述示例中，`y` 是一个形状为 **(6, )** 的一维张量。通过使用 `y.reshape((-1, 1))`，我们将其重塑为一个形状为 **(6, 1)** 的列向量。重塑后的张量 `reshaped_y` 按列排列原始张量 `y` 的元素。

2. `y.reshape(-1)`：一维张量
将张量 `y` 重塑为一个扁平化的一维张量。
```python
     import torch

     y = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
     reshaped_y = y.reshape(-1)

     print(reshaped_y)
     print(reshaped_y.shape)
     
     输出结果：
  
     tensor([1, 2, 3, 4, 5, 6])
     torch.Size([6])
     ```
在上述示例中，`y` 是一个形状为 `(2, 3)` 的二维张量。通过使用 `y.reshape(-1)`，我们将其重塑为一个扁平化的一维张量。重塑后的张量 `reshaped_y` 将原始张量 `y` 的元素按行优先的顺序展开。

总结：
- `y.reshape((-1, 1))` 将张量 `y` 重塑为一个列向量，**维度**为 (num_elements, 1)，其中 `num_elements` 是原始张量 `y` 的元素数量。
- `y.reshape(-1)` 将张量 `y` 重塑为一个扁平化的一维张量，**维度**为 (num_elements,)，其中 `num_elements` 是原始张量 `y` 的元素数量。

### 二维张量和一位张量的区别
带有维度的二维张量，而非一维张量
```python
     import torch

     y = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
     reshaped_y = y.reshape(-1, 6)

     print(reshaped_y)
     print(reshaped_y.shape)
     
     输出结果：
  
	tensor([[1, 2, 3, 4, 5, 6]])
	torch.Size([1, 6])
     ```