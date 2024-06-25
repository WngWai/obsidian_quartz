在Python中，`random.shuffle()`函数用于**打乱列表的顺序**。会修改原始列表 `x`，将其元素随机排序。
```python
random.shuffle(x, random=random.random)
```
**参数**：
- `x`：要**打乱顺序的列表**。
- `random`：可选参数，用于指定自定义的随机数生成器函数。默认值为`random.random`，即使用Python内置的随机数生成器。
**示例**：
```python
import random

# 创建一个列表
my_list = [1, 2, 3, 4, 5]

# 使用random.shuffle()打乱列表顺序
random.shuffle(my_list)

# 查看打乱后的列表
print(my_list)
```

**输出**：
```
[2, 4, 3, 5, 1]
```

在上述示例中，我们创建了一个列表 `my_list`，其中包含了整数1到5。

然后，我们调用 `random.shuffle(my_list)` 来打乱列表的顺序。`random.shuffle()`函数会修改原始列表 `my_list`，将其元素随机排序。

最后，我们打印输出打乱后的列表 `my_list`，可以看到列表元素的顺序已经被随机打乱了。

需要注意的是，`random.shuffle()`函数会直接修改原始列表，并不会返回新的列表。因此，如果需要保留原始列表的顺序，可以在打乱之前创建列表的副本进行操作。

`random.shuffle()`函数通常用于需要随机化数据顺序的场景，例如在机器学习中对训练数据进行随机批次处理，或者在游戏开发中对游戏元素进行随机排序等。