在R语言中，`purrr`包中的`partial()`函数用于**创建一个部分应用了部分参数的新函数**。部分应用是指在函数调用中固定一部分参数的值，然后返回一个接受剩余参数的新函数。
**函数定义**：
```R
partial(.f, ...)
```
**参数**：
- `.f`：要部分应用的函数。可以是函数名或lambda表达式。
- `...`：要部分应用的参数。可以是具体的值或使用`~`定义的参数。
**示例**：
```R
library(purrr)

# 示例1：创建部分应用的平方函数
square <- partial(`^`, 2)

# 调用部分应用的平方函数
result <- square(3)

# 打印结果
print(result)
# 输出: 9


# 示例2：创建部分应用的字符串连接函数
paste_hello <- partial(paste, "Hello")

# 调用部分应用的字符串连接函数
result <- paste_hello("World")

# 打印结果
print(result)
# 输出: "Hello World"


# 示例3：创建部分应用的lambda表达式函数
add_numbers <- partial(~ .x + .y, .x = 10)

# 调用部分应用的lambda表达式函数
result <- add_numbers(5)

# 打印结果
print(result)
# 输出: 15
```

在示例1中，我们使用`partial()`函数创建了一个部分应用的平方函数。我们传递了`^`函数和固定的参数2给`partial()`函数，返回一个新函数`square`。然后，我们调用部分应用的平方函数`square`，传递参数3，计算结果为9。

在示例2中，我们使用`partial()`函数创建了一个部分应用的字符串连接函数。我们传递了`paste`函数和固定的参数"Hello"给`partial()`函数，返回一个新函数`paste_hello`。然后，我们调用部分应用的字符串连接函数`paste_hello`，传递参数"World"，结果为"Hello World"。

在示例3中，我们使用`partial()`函数创建了一个部分应用的lambda表达式函数。我们使用lambda表达式`~ .x + .y`作为参数，并通过`.x = 10`来固定其中的参数`x`为10，返回一个新函数`add_numbers`。然后，我们调用部分应用的lambda表达式函数`add_numbers`，传递参数5，计算结果为15。

通过使用`purrr`包中的`partial()`函数，我们可以方便地创建部分应用的新函数，固定部分参数的值，简化函数调用，并提高代码的可读性和重用性。