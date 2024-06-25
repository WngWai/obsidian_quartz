是Python中的一个重要概念，它允许一个函数在其定义时捕捉和绑定到其作用域内的变量，即使在函数被调用之后，这些变量依然可以访问。闭包本质上是一个函数对象，它记录了它的环境（即它在创建时的上下文）。

Python闭包（Closure）详解 - 风影忍着的文章 - 知乎
https://zhuanlan.zhihu.com/p/453787908

`嵌套在函数里的函数，函数可以访问外部函数的局部变量。`

因为函数内的变量均为**局部变量**，为了能使用函数里的局部变量，通过闭包（嵌套子函数）实现调用！
```python
def f1():
    n=999;

print(n) 
# 输出：NameError

def f1():
    n=999
    def f2():
        print(n)    
f1() 
# 输出：999
    
```

*剩下的其实不用看了！*

### 1. 闭包的定义

闭包有以下三个特征：
1. 闭包是一个嵌套函数——一个**定义在另一个函数内部的函数**。
2. 闭包可以访问其外部函数的变量。
3. 外部函数返回内部函数，内部函数可以记住和访问其定义时的上下文环境。



### 2. 闭包的使用

闭包通常用于在函数内部创建和返回函数对象，并且这些函数对象**可以访问其外部函数的局部变量**。以下是一个简单的闭包示例：

```python
def outer_func(x):
    def inner_func(y):
        return x + y
    return inner_func

closure = outer_func(10)
print(closure(5))  # 输出: 15
```

在这个例子中，`inner_func` 是一个闭包，它捕获了外部函数 `outer_func` 的变量 `x`。即使 `outer_func` 已经返回，`inner_func` 依然可以访问 `x`。

### 3. 闭包的应用

闭包在Python中有多种用途，包括但不限于以下几种：

- **保持状态**：闭包可以用来记住函数被调用时的状态信息。
- **避免全局变量**：闭包可以避免使用全局变量，从而使代码更加模块化和可维护。
- **装饰器**：闭包常用于实现装饰器，装饰器是修改函数行为的一种设计模式。
- **工厂函数**：闭包可以用来创建带有特定参数的函数。

### 4. 按功能分类的闭包示例

#### 4.1 保持状态

闭包可以记住并更新其环境中的状态：

```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

counter1 = make_counter()
print(counter1())  # 输出: 1
print(counter1())  # 输出: 2
print(counter1())  # 输出: 3
```

#### 4.2 避免全局变量

闭包可以避免使用全局变量，从而使代码更加模块化：

```python
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))  # 输出: 10
print(triple(5))  # 输出: 15
```

#### 4.3 实现装饰器

装饰器本质上就是一个返回闭包的高阶函数，用于增强函数的功能：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello('Alice')
# 输出:
# Something is happening before the function is called.
# Hello, Alice!
# Something is happening after the function is called.
```

#### 4.4 工厂函数

闭包可以用来创建带有特定参数的函数：

```python
def power_factory(exp):
    def power(base):
        return base ** exp
    return power

square = power_factory(2)
cube = power_factory(3)

print(square(4))  # 输出: 16
print(cube(2))    # 输出: 8
```

### 小结

闭包在Python中是一个强大而灵活的工具，具有以下优点：
- **模块化**：通过闭包，可以避免使用全局变量，使代码更加模块化和可维护。
- **保持状态**：闭包可以记住并维护其环境中的状态信息，适用于需要记住函数调用状态的场景。
- **装饰器**：闭包是实现装饰器的核心机制，装饰器用于增强函数的功能。
- **工厂函数**：闭包可以用作工厂函数，创建具有特定参数的函数。

了解并巧妙使用闭包可以显著提高代码的灵活性和可读性。