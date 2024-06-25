 `random` 模块是用于**生成随机数**的标准库。它提供了各种函数来生成不同类型的随机数，以及执行与随机性相关的操作。以下是一些 `random` 模块中常用的函数：

- `random.random()`: 生成一个 0 到 1 之间的**随机浮点数**。
- `random.randint(a, b)`: 生成一个在**指定范围内的整数**，包括起始值 `a` 和结束值 `b`。
- `random.choice(sequence)`: 从序列中（如列表或字符串）随机选择一个元素。
- `random.shuffle(sequence)`: 随机**打乱一个序列中的元素的顺序**。
- `random.sample(sequence, k)`: 从序列中随机选择 `k` 个元素，返回一个新的列表。

以下是一个使用 `random` 模块生成随机数的示例代码：

```python
import random

# 生成一个 0 到 1 之间的随机浮点数
random_num = random.random()
print(random_num)

# 生成一个在指定范围内的整数
random_int = random.randint(1, 10)
print(random_int)

# 从列表中随机选择一个元素
my_list = [1, 2, 3, 4, 5]
random_choice = random.choice(my_list)
print(random_choice)

# 随机打乱列表中的元素顺序
random.shuffle(my_list)
print(my_list)

# 从列表中随机选择两个元素
random_sample = random.sample(my_list, 2)
print(random_sample)
```

这只是 `random` 模块中一些常用函数的简单示例。该模块还提供了其他更多有用的函数，如生成随机字母、随机高斯分布数等。
