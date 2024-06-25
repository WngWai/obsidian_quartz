`apply()` 方法可以对 DataFrame 或 Series 中的**每个元素**应用一个函数。这个函数可以是一个自定义函数，也可以是一个匿名函数（lambda 函数）。
lambda 函数是一种匿名函数，通常用于一次性的、简单的函数应用场景。它的语法形式为：

```python
lambda arguments: expression
```

其中，`arguments` 是**参数列表**，可以包含多个参数，用逗号分隔，用到DF中的每个元素值；`expression` 是一个表达式，是**函数的返回值**。

`将分组后的小组数据当作单独的df数据框，进行apply处理。基本操作单位是DataFrame，而非,group_by的df的apply的基本操作单位是Series。

```python
def get_oldest_staff(x):
     df = x.sort_values(by = 'age',ascending=True)
     return df.iloc[-1,:]

​
 oldest_staff = data.groupby('company',as_index=False).apply(get_oldest_staff)
​
 oldest_staff

# 输出
  company  salary  age  
0       A      23   33       
1       B      21   40       
2       C      43   35 
```

![[Pasted image 20231213111517.png]]


在 `apply()` 方法中使用 lambda 函数时，我们可以将函数作为参数传递给 `apply()` 方法，也可以直接在 `apply()` 方法中使用 lambda 函数。例如：

```python
import pandas as pd

# 创建数据
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 使用自定义函数
def my_func(x):
    return x + 1

# 对每个元素应用函数
result1 = data.apply(my_func)
result2 = data.apply(lambda x: x + 1)

print(result1)
print(result2)
```

在上述代码中，我们首先创建了一个包含两列的 DataFrame。然后，我们定义了一个自定义函数 `my_func()`，该函数的功能是**将传入的参数加 1**。接着，我们使用 `apply()` 方法分别对 **DataFrame 中的每个元素**应用了 `my_func()` 函数和一个 lambda 函数，将每个元素加 1。最后，我们打印了两个结果。
上面代码的输出结果为：

```python
   A  B
0  2  5
1  3  6
2  4  7
   A  B
0  2  5
1  3  6
2  4  7
```

这两个结果是相同的，因为它们都对 DataFrame 中的每个元素应用了相同的函数。需要注意的是，lambda 函数通常用于简单的操作，如果需要进行复杂的操作，建议使用自定义函数。



### [[py_lambda]]定义匿名函数
Lambda函数是一种匿名函数，也称为“轻量级函数”，它可以在一行代码中定义一个函数，非常适合于需要临时定义函数的场景。

Lambda函数的语法格式如下：

```python
lambda argument_list: expression
```

其中，`argument_list`是参数列表，可以包含零个或多个参数，多个参数之间用逗号分隔；`expression`是函数体，可以是任何有效的 Python 表达式，通常用于计算和返回结果。

Lambda函数通常用于作为其他函数的参数，例如在 `map()`、`filter()`、`reduce()` 等函数中使用。

下面是一个简单的例子，演示如何使用Lambda函数将一个列表中的元素平方并返回一个新的列表：

```python
lst = [1, 2, 3, 4, 5]
new_lst = list(map(lambda x: x ** 2, lst))
print(new_lst)
```

输出结果为：

```python
[1, 4, 9, 16, 25]
```

在这个例子中，我们使用 `map()` 函数将一个 Lambda 函数应用到列表 `lst` 中的**每一个元素上**，Lambda 函数将每个元素平方并返回结果，最终得到一个新的列表 `new_lst`。