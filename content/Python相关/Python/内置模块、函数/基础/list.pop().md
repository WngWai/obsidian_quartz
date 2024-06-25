在Python中，`pop()` 方法是列表（list）的一个**内置方法**，用于移除列表中的一个元素，并返回该元素的值。默认情况下，`pop()` 移除并返回列表的最后一个元素，但你可以通过传入索引参数来指定要移除的元素。
### 函数定义：
```python
list.pop(index=-1)
```
### 参数介绍：
- **index** (可选)：要移除的元素的索引。**默认值**为 `-1`，表示移除并返回列表的最后一个元素。如果你传入一个正数索引，`pop()` 将会移除并返回该索引处的元素。如果你传入一个负数索引，`pop()` 将会从列表的末尾开始计数，移除并返回相应的元素。

也可以指定正常的索引值！如list_.pop(1)返回第二个元素，并剔除相应值！


### 举例：
```python
# 创建一个列表
my_list = [1, 2, 3, 4, 5]
# 移除并返回列表的最后一个元素
last_element = my_list.pop()
print(last_element)  # 输出: 5
print(my_list)       # 输出: [1, 2, 3, 4]
# 移除并返回索引为1的元素
element_at_index_1 = my_list.pop(1)
print(element_at_index_1)  # 输出: 2
print(my_list)             # 输出: [1, 3, 4]
# 移除并返回索引为-2的元素（从列表末尾开始计数）
element_at_index_minus_2 = my_list.pop(-2)
print(element_at_index_minus_2)  # 输出: 3
print(my_list)                   # 输出: [1, 4]
```
如果你尝试使用 `pop()` 方法移除一个不存在的索引处的元素，将会引发 `IndexError` 异常。例如：
```python
# 尝试移除索引为5的元素，但列表只有两个元素
my_list.pop(5)  # 引发 IndexError: pop index out of range
```
在实际使用中，`pop()` 方法常用于需要临时存储并最终移除元素的场景，或者需要在列表中改变元素顺序的情况。
