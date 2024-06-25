`filter()` 是 Python 内置函数之一，用于**过滤可迭代对象（如列表、元组等）中的元素，返回满足指定条件的元素的迭代器。**
**定义：**
```python
filter(function, iterable)
```

### 参数介绍：

- **`function`：** 一个函数，用于定义过滤条件，返回值为布尔类型。

- **`iterable`：** 一个可迭代对象，包含要过滤的元素。

### 示例：

```python
# 定义一个函数，用于筛选偶数
def is_even(x):
    return x % 2 == 0

# 创建一个列表
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用 filter() 过滤出偶数
even_numbers = filter(is_even, numbers)

# 将结果转换为列表
result = list(even_numbers)

# 输出结果
print(result)  # 输出: [2, 4, 6, 8, 10]
```

### 注意事项：

- `filter()` 函数返回的是一个迭代器，因此需要将其转换为列表或其他类型的可迭代对象以查看结果。

- `function` 函数应该返回布尔值，`True` 表示保留元素，`False` 表示过滤掉元素。

- `filter()` 函数只保留满足条件的元素，不对元素进行任何操作，类似于一个筛选器。

- 可以使用匿名函数（`lambda`）来定义过滤条件：

  ```python
  numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  even_numbers = filter(lambda x: x % 2 == 0, numbers)
  result = list(even_numbers)
  print(result)  # 输出: [2, 4, 6, 8, 10]
  ```

`filter()` 函数在许多情况下非常有用，特别是在需要根据某个条件从列表中提取特定元素时。


