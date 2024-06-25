`lambda` [ˈlæmdə]是 Python 中用于创建匿名函数的关键字。它允许你定义简单的、单行的函数，通常用于一些简单的操作，而不必显式地定义一个完整的函数。`lambda` 函数的语法如下：

```python
lambda arguments: expression
```

- **`lambda`：** 关键字，表示创建一个 lambda 函数。

- **`arguments`：** 输入参数，类似于函数的**参数列表**。

如果是字典也是代表匿名函数的参数，实际指列表中的每个元素，这里是指**列表中的字典**！
students = [
    {"name": "John", "grade": 90},
    {"name": "Jane", "grade": 88},
    {"name": "Dave", "grade": 92},
]
sorted_students = sorted(students, key=lambda student1: student1['grade'])
print(sorted_students)


- **`expression`：** **表达式**，是**对输入参数的操作**，也是 **lambda 函数的返回值**。

### 示例：

以下是一个简单的使用 lambda 函数的例子：

```python
# 使用 lambda 创建一个加法函数
add = lambda x, y: x + y

# 调用 lambda 函数
result = add(3, 5)

# 输出结果
print(result)  # 输出: 8
```

在这个例子中，`lambda x, y: x + y` 创建了一个接受两个参数 `x` 和 `y` 的 lambda 函数，执行的操作是返回它们的和。然后，通过 `add(3, 5)` 调用这个 lambda 函数，得到结果 `8`。

### 注意事项：

- Lambda 函数通常用于**定义简单的功能，而不是复杂的逻辑**。对于复杂的函数，通常建议使用常规的 `def` 关键字定义函数。

- Lambda 函数是**匿名**的，因此它们通常在需要一个小型功能的地方使用，而不是在整个程序中定义。

- Lambda 函数可以用于许多 Python 内置的函数和方法，如 `map()`、`filter()` 等，以提供一种简洁的语法。例如：

  ```python
  numbers = [1, 2, 3, 4, 5]
  squared = list(map(lambda x: x**2, numbers))
  print(squared)  # 输出: [1, 4, 9, 16, 25]
  ```

上述示例中，`lambda` 用于创建一个平方函数，并将其应用于列表中的每个元素。