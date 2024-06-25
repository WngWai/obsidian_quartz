在Python中，`sorted()` 是一个内置函数，用于从任何**可迭代的系列（列表、元组、字符串等）生成一个新的排序后的列表**。与 `list.sort()` 方法不同，`sorted()` 函数不会修改原始数据，而是返回一个全新的排序后的列表。


```python
sorted(iterable, key=None, reverse=False)
```

### 主要参数介绍：

- `iterable`: 这是一个可迭代的对象，如列表、元组、字符串等，你想要对其进行排序。

- `key`: 这是一个函数，用于从每个元素中提取一个用于比较的键（类似于**排序的标准**）。这是一个可选参数，默认为 `None`。

- `reverse`: 这是一个布尔值，用于指定排序应该是升序还是降序。默认值是 `False`，表示排序结果为升序。 True，则返回结果为降序。

### 应用举例：

1. 使用 `sorted()` 对列表进行排序：
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # 输出: [1, 1, 2, 3, 4, 5, 6, 9]
```

2. 使用 `reverse` 参数对列表进行降序排序：
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers, reverse=True)
print(sorted_numbers)  # 输出: [9, 6, 5, 4, 3, 2, 1, 1]
```

5. 使用 `sorted()` 对字符串进行排序：
```python
s = "hello"
sorted_s = sorted(s)
print(sorted_s)  # 输出: ['e', 'h', 'l', 'l', 'o']
print("".join(sorted_s))  # 将排序后的字符列表组合成字符串: 'ehllo'
```

3. 使用 `key` 参数对字符串列表按长度排序：
```python
words = ["banana", "apple", "cherry", "date"]
sorted_words = sorted(words, key=len)
print(sorted_words)  # 输出: ['date', 'apple', 'banana', 'cherry']
```

4. 使用 `key` 参数和 lambda 表达式对包含字典的列表进行排序：
这个student是匿名函数的参数，接收的是列表中的字典元素！
```python
students = [
    {"name": "John", "grade": 90},
    {"name": "Jane", "grade": 88},
    {"name": "Dave", "grade": 92},
]
sorted_students = sorted(students, key=lambda student: student['grade'])
print(sorted_students)
# 输出: [{'name': 'Jane', 'grade': 88}, {'name': 'John', 'grade': 90}, {'name': 'Dave', 'grade': 92}]
```

直接对字典进行排序：
```python
# 使用 sorted() 函数根据字典值进行降序排序
sorted_students = sorted(students.items(), key=lambda x: x[1], reverse=True)

# 打印排序后的字典
print(sorted_students)

[('David', 95), ('Bob', 92), ('Alice', 88), ('Charlie', 83)]
```


`sorted()` 函数是非常灵活的，可以用于几乎任何可排序的数据类型，并且总是返回一个排序后的列表，不影响原数据。