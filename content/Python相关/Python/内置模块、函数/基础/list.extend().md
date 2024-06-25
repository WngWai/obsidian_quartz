`list.extend()` 是 Python 中内置的列表（list）数据类型的一个方法。该方法用于**将一个迭代器（如列表、元组、集合等）中的所有元素添加到当前列表的末尾**。
### 参数
- `iterable`：这是一个可迭代的对象，例如列表、元组、集合等。`extend()` 方法会将这个可迭代对象中的所有元素逐一添加到当前列表中。
### 返回值
`list.extend()` 方法没有返回值，它直接修改原列表。
### 应用举例
```python
# 创建一个空列表
my_list = []
# 使用 extend() 方法添加元素
my_list.extend([1, 2, 3])
my_list.extend((4, 5, 6))  # 元组也可以
my_list.extend(set([7, 8, 9]))  # 集合也可以
# 输出修改后的列表
print(my_list)  # 输出: [1, 2, 3, 4, 5, 6, 7, 8, 9]
# 可以添加任何可迭代的对象
my_list.extend("hello")
my_list.extend(range(3))
# 输出修改后的列表
print(my_list)  # 输出: [1, 2, 3, 4, 5, 6, 7, 8, 9, 'h', 'e', 'l', 'l', 'o', 0, 1, 2]
```
### 注意事项
- `extend()` 方法不会创建一个新的列表，而是修改当前列表。
- 如果 `iterable` 参数不是一个迭代器，而是单个元素（如整数、字符串等），则会引发 `TypeError`。
- `extend()` 方法可以接受任何可迭代的对象，不仅限于列表。
```python
# 错误的用法：试图将单个元素添加到列表中
my_list.extend(10)  # TypeError: 'int' object is not iterable
```
`list.extend()` 是一个非常常用的方法，用于在 Python 编程中快速地将一个迭代器的所有元素合并到一个列表中。
