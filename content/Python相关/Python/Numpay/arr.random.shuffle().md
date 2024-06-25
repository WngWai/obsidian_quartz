在Python中，`np.random.shuffle()`函数用于**随机打乱数组或列表中的元素顺序**。

**函数定义**：
```python
np.random.shuffle(x)
```

**参数**：
以下是`np.random.shuffle()`函数中的参数：

- `x`：要打乱顺序的数组或列表。

**示例**：
以下是使用`np.random.shuffle()`函数打乱数组或列表顺序的示例：

```python
import numpy as np

# 示例1：打乱数组顺序
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(arr)  # 输出: [3 4 2 5 1]

# 示例2：打乱列表顺序
lst = [1, 2, 3, 4, 5]
np.random.shuffle(lst)
print(lst)  # 输出: [4, 1, 2, 5, 3]
```

在上述示例中，我们首先导入了`numpy`库。

在示例1中，我们创建了一个数组`arr`，其中包含了一些整数。然后，我们使用`np.random.shuffle()`函数打乱了数组`arr`的顺序，并将结果保存回`arr`。最终输出的结果是打乱后的数组。

在示例2中，我们创建了一个列表`lst`，其中包含了一些整数。然后，我们使用`np.random.shuffle()`函数打乱了列表`lst`的顺序，并将结果保存回`lst`。最终输出的结果是打乱后的列表。

请注意，`np.random.shuffle()`函数会直接修改原始的数组或列表，而不会创建一个新的副本。

`np.random.shuffle()`函数通过随机重新排列数组或列表中的元素来实现打乱顺序的效果。它使用了随机种子，以确保每次运行结果的随机性。

以上是`np.random.shuffle()`函数的基本用法和示例。它在数据处理、随机采样等场景中常用于打乱数据的顺序。