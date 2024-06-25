`map()` 函数概述：
![[Pasted image 20231213153410.png]]

![[Pasted image 20231213153416.png]]

https://zhuanlan.zhihu.com/p/100064394

**功能：** `map()` 是 Python 内置函数之一，用于**对可迭代对象（如列表、元组等）中的每个元素应用指定的函数**，返回一个包含结果的迭代器。

**所属包：** 内置函数，无需导入额外的包。

**定义：**
```python
map(function, iterable, ...)
```

### 参数介绍：

- **`function`：** 一个函数，用于对每个元素进行操作。

- **`iterable`：** 一个或多个可迭代对象，包含要处理的元素。

### 示例：

```python
# 定义一个函数，用于计算平方
def square(x):
    return x**2

# 创建一个列表
numbers = [1, 2, 3, 4, 5]

# 使用 map() 对列表中的每个元素应用平方函数
squared = map(square, numbers)

# 将结果转换为列表
result = list(squared)

# 输出结果
print(result)  # 输出: [1, 4, 9, 16, 25]
```

### 注意事项：

- `map()` 函数返回的是一个迭代器，因此需要将其转换为列表或其他类型的可迭代对象以查看结果。

- `map()` 函数可以接受多个可迭代对象，此时传递的函数必须接受与传递的可迭代对象相同数量的参数。

```python
# 同时对两个列表中的元素求和
numbers1 = [1, 2, 3, 4]
numbers2 = [5, 6, 7, 8]

sum_result = map(lambda x, y: x + y, numbers1, numbers2)
result = list(sum_result)
print(result)  # 输出: [6, 8, 10, 12]
```

在上述示例中，`lambda` 函数接受两个参数，分别对两个列表中的对应元素求和。