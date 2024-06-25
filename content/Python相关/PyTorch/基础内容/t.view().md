在PyTorch中，`view()`函数是一个Tensor方法，用于重新塑形张量而不改变其数据。这个函数的功能类似于NumPy的`reshape`方法，但`view()`专门用于`torch.Tensor`对象。该方法返回的新张量与原张量共享数据，因此对新张量的修改也会影响原张量，反之亦然。


```python
Tensor.view(*shape)
t.view(1,3,28,28)
t.view((-1,3,28,28))
```


- **shape**：目标形状。可以是一系列的整数，表示想要的输出形状。特别地，你可以使用`-1`作为某个维度，PyTorch将自动计算这个维度的大小。

### 注意事项

在使用`.view()`方法之前，要确保Tensor是在**内存中连续**的。如果Tensor不是连续的，你可能需要先调用`.contiguous()`方法。例如：`tensor.contiguous().view(-1, 24)`。

```python
t.is_contiguous()检查张量是否连续，返回布尔值
t.contiguous()跟非连续张量创建连续张量
```



### 常见应用举例

#### 1. 改变Tensor形状

假设你有一个形状为`(4, 4)`的Tensor，你想要将其改变形状为`(2, 8)`。

```python
import torch

x = torch.arange(16).view(4, 4)  # 创建一个形状为 [4, 4] 的Tensor
y = x.view(2, 8)  # 将x的形状改变为 [2, 8]
print("Original Tensor:\n", x)
print("Reshaped Tensor:\n", y)
```

#### 2. 展平Tensor

在处理图像或进行批处理前，经常需要将多维度的数据展平为一维。使用`-1`参数可以轻松完成这项工作。

```python
import torch

x = torch.randn(4, 4)  # 假设这是一个 [4, 4] 形状的Tensor
y = x.view(-1)  # 将x展平为一维
print("Original Tensor:\n", x)
print("Flattened Tensor:\n", y)
```

#### 3. 将批量图像数据展平为二维

当你在处理一个批量的图像数据时（例如形状为`(batch_size, channels, height, width)`），有时你可能需要将每个图像展平为一维，但保留批次维度不变。

```python
import torch

batch_size = 10
channels = 3
height = 32
width = 32

# 创建一个模拟的批量图像数据 [batch_size, channels, height, width]
x = torch.randn(batch_size, channels, height, width)

# 将每张图片展平为一维向量，同时保留批次维度
y = x.view(batch_size, -1)  # y 的形状将为 [batch_size, channels*height*width]

print("Original Tensor Shape:", x.shape)
print("Reshaped Tensor Shape:", y.shape)
```

这些例子展示了`view()`方法在数据预处理和模型构建时的不同应用场景。通过更改Tensor的形状，可以轻松地适应不同的操作和网络结构需求。