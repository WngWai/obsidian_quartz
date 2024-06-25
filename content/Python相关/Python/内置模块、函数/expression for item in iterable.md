`expression for item in iterable` 是一种**列表推导式**（List Comprehension）的语法形式，在 Python 中用于**快速生成列表**。它可以根据迭代的元素，**在每个列表元素上应用一个表达式（express）**，并**将结果收集到一个新的列表中**。
express
```python
new_list = [expression for item in iterable if condition]
```

- `expression` 是对现有元素的操作，可以是一个表达式、函数调用或任何可返回值的操作。

- `item` 是来自于 `iterable` 的每个元素，在循环迭代过程中逐个取值。

- `iterable` 是可迭代的对象，如列表、元组、集合、字符串等。

- `condition` 循环中可用于**过滤元素**，只有满足条件的元素才会被包含在新列表中。


1. 将列表中的每个元素平方后生成新的列表：

```python
old_list = [1, 2, 3, 4, 5]
squared_list = [x ** 2 for x in old_list]
print(squared_list)  # 输出: [1, 4, 9, 16, 25]
```

2. 使用条件过滤列表中的偶数：

```python
old_list = [1, 2, 3, 4, 5]
even_list = [x for x in old_list if x % 2 == 0]
print(even_list)  # 输出: [2, 4]
```

3. 将字符串列表中的字符串转换为大写：

```python
old_list = ['apple', 'banana', 'cherry']
upper_list = [fruit.upper() for fruit in old_list]
print(upper_list)  # 输出: ['APPLE', 'BANANA', 'CHERRY']
```

4. 使用嵌套列表推导式创建二维列表：

```python
matrix = [[i + j for j in range(3)] for i in range(3)]
print(matrix)  # 输出: [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

列表推导式可以简化代码并提高可读性，但要注意不要过度使用，以免降低代码的可读性。


每个表达式作为列表元素，合并形成一个列表！如果元素本身就是列表，合并在一起就是一个高维矩阵了！
```python
list_ = [list(map(int, input().split(','))) for _ in range(n)]
```