在PyTorch中，`nn.Flatten()`函数用于**将多维输入张量展平为一维张量**。

![[Pasted image 20240424083936.png|400]]

**函数定义**：
```python
nn.Flatten(start_dim=1, end_dim=-1)
```

**参数**：
以下是`nn.Flatten()`函数中的参数：

- `start_dim`：可选参数，指定从哪个维度开始展平输入张量。**默认值为1**，表示**从第二个维度开始展平**。
如 256\*28\*28，第1个维度是256。从28开始展平

- `end_dim`：可选参数，指定展平结束的维度。**默认值为-1**，表示展平到**最后一个维度**。
28\*28=784，256\*784，就理解为**二维矩阵数据变为一维向量数据**！

**示例**：
以下是使用`nn.Flatten()`函数展平输入张量的示例：

```python
import torch
import torch.nn as nn

# 创建一个输入张量
input_tensor = torch.randn(64, 3, 32, 32)

# 创建Flatten层
flatten = nn.Flatten()

# 将输入张量展平
output = flatten(input_tensor)

print(input_tensor.size())  # 输出: torch.Size([64, 3, 32, 32])
print(output.size())  # 输出: torch.Size([64, 3072])
```

在上述示例中，我们首先导入了所需的模块和类。

然后，我们创建了一个随机初始化的输入张量`input_tensor`，其形状为`[64, 3, 32, 32]`，表示64个3通道的32x32图像。

接下来，我们使用`nn.Flatten()`函数创建了一个`flatten`层。

最后，我们将输入张量`input_tensor`作为`flatten`层的输入，并将输出保存在`output`变量中。输出张量的形状为`[64, 3072]`，表示64个样本的3072维一维张量。

`nn.Flatten()`函数将输入张量在指定的维度范围内展平。在示例中，由于没有提供`start_dim`和`end_dim`参数，所以默认将第二个维度到最后一个维度进行展平。

请注意，`start_dim`和`end_dim`参数可根据实际情况进行调整以满足不同的展平需求。

以上是`nn.Flatten()`函数的基本用法和示例。更多详细的参数和选项可以参考PyTorch的官方文档。