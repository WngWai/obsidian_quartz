在Python中，`isinstance()`函数用于检查一个对象是否属于指定的类型。
**函数定义**：
```python
isinstance(object, classinfo)
```
**参数**：
- `object`：要检查的对象。
- `classinfo`：可以是一个类型对象或一个类型元组（由多个类型对象组成）。如果`object`的类型与`classinfo`中的**任何一个类型相同**，则返回True。如果`classinfo`是个元组，其中包含多个类型对象，则只要`object`的类型与其中任何一个类型相同，就返回True。
**示例**：
```python
# 示例：检查对象类型
x = 5
y = "Hello"
z = [1, 2, 3]

# 检查x的类型
print(isinstance(x, int))  # True
print(isinstance(x, str))  # False
print(isinstance(x, (int, float)))  # True

# 检查y的类型
print(isinstance(y, str))  # True
print(isinstance(y, (int, float)))  # False

# 检查z的类型
print(isinstance(z, list))  # True
print(isinstance(z, (tuple, dict)))  # False
```

在示例中，我们首先创建了三个对象：`x`是一个整数，`y`是一个字符串，`z`是一个列表。

然后，我们使用`isinstance()`函数来检查这些对象的类型。对于`x`，我们分别检查它是否是整数、字符串，以及整数或浮点数的组合。对于`y`，我们检查它是否是字符串，以及整数或浮点数的组合。对于`z`，我们检查它是否是列表，以及元组或字典的组合。

根据上述示例，`isinstance()`函数的返回结果将根据对象的类型和提供的类型信息进行判断，如果对象的类型与给定的类型信息相符，则返回True，否则返回False。

使用`isinstance()`函数可以帮助我们在编程中进行类型检查，以便根据不同类型的对象执行不同的操作或逻辑。