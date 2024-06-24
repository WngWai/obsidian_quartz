在R语言中，`purrr`包中的`safely()`函数用于创建一个**安全执行函数**，该函数会捕获潜在的错误并返回一个包含错误信息的结果。这对于处理可能会引发错误的操作非常有用，以防止程序中断并提供错误处理机制。
**函数定义**：
```R
safely(.f, otherwise = NULL)
```
**参数**：
- `.f`：要安全执行的函数。
- `otherwise`：当函数执行错误时返回的默认值。默认为`NULL`。
**示例**：
```R
library(purrr)

# 示例1：安全执行除法操作
safe_divide <- safely(function(x, y) {
  x / y
})

# 安全执行除法操作
result <- safe_divide(10, 2)
if (is.null(result$error)) {
  print(result$result)
} else {
  print("Error occurred!")
}
# 输出: 5

result <- safe_divide(10, 0)
if (is.null(result$error)) {
  print(result$result)
} else {
  print("Error occurred!")
}
# 输出: "Error occurred!"


# 示例2：指定默认返回值
safe_sqrt <- safely(sqrt, otherwise = "NaN")

# 安全执行平方根操作
result <- safe_sqrt(9)
if (is.null(result$error)) {
  print(result$result)
} else {
  print("Error occurred!")
}
# 输出: 3

result <- safe_sqrt(-9)
if (is.null(result$error)) {
  print(result$result)
} else {
  print("Error occurred!")
}
# 输出: "NaN"
```

在示例1中，我们使用`safely()`函数创建了一个安全执行函数`safe_divide()`，该函数用于进行除法操作。然后，我们调用`safe_divide()`函数进行除法操作，传递参数`10`和`2`进行除法计算。如果计算成功，我们打印结果`5`；如果计算出错，我们打印错误消息"Error occurred!"。接着，我们再次调用`safe_divide()`函数进行除法操作，但这次将第二个参数设为`0`，这会导致除法错误。然后，我们根据是否存在错误来打印不同的消息。

在示例2中，我们使用`safely()`函数创建了一个安全执行函数`safe_sqrt()`，该函数用于进行平方根操作。然后，我们调用`safe_sqrt()`函数进行平方根计算，传递参数`9`进行计算。如果计算成功，我们打印结果`3`；如果计算出错，我们打印错误消息"Error occurred!"。接着，我们再次调用`safe_sqrt()`函数进行平方根计算，但这次将参数设为`-9`，这会导致计算出错。然后，我们根据是否存在错误来打印不同的消息，并返回默认值"NaN"。

通过使用`purrr`包中的`safely()`函数，我们可以创建安全执行函数，以处理可能会引发错误的操作。这可以帮助我们在程序中避免崩溃并提供错误处理机制，确保代码的鲁棒性和稳定性。