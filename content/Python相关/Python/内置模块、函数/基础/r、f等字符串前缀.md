在Python中，字符串前面加上`r`或`f`表示对字符串进行特殊的处理。改变字符串的解释方式的前缀
### `r`字符串（Raw字符串）
**忽略转义符**，它将被视为**原始字符**串，并且**不会进行转义**。这意味着字符串中的转义字符（例如`\n`、`\t`等）将被视为普通字符而不会被解释。
```python
path = r'C:\Users\username\Documents'
print(path)  # 'C:\Users\username\Documents'
```
在上述示例中，字符串`path`前面加了`r`，所以其中的反斜杠`\`被当作普通字符处理，不会被解释为转义字符。
### f字符串（F-strings）
格式化字符串，在这种字符串中，可以包含花括号`{}`，并在其中**插入变量或表达式**的值。
```python
name = "Alice"
age = 25
message = f"My name is {name} and I am {age} years old."
print(message)  # 'My name is Alice and I am 25 years old.'
```
在上述示例中，字符串`message`前面加了`f`，并使用花括号`{}`插入了变量`name`和`age`的值。在字符串中，这些花括号将被替换为相应的变量值。
### `r`和`f`同时使用
创建既是**原始字符串**又是**格式化字符串**的字符串。例如，`rf`字符串类似于普通字符串，但支持在其中插入表达式的值。
```python
value = 10
message = rf"Value: {value + 5}"
print(message)  # 'Value: 15'
```
在上述示例中，字符串`message`使用了`rf`前缀，其中的表达式`value + 5`将被求值并插入字符串中。
### `b`字符串（Bytes字符串）
当字符串前面加上`b`时，它将被视为**字节字符串**，即一系列原始字节序列。字节字符串在处理二进制数据或与底层操作系统交互时非常有用。
```python
data = b'\x48\x65\x6c\x6c\x6f'  # 字节序列表示字符串 "Hello"
print(data.decode())  # 'Hello'
```
### `u`字符串（Unicode字符串）
在Python 2.x中使用，表示**Unicode字符串**。在Python 3中，默认字符串就是Unicode字符串，所以不再需要显式声明。
### `''' '''`或`""" """`（三重引号字符串）
用于创建**多行字符串**。三重引号可以跨越多行，并且可以包含换行符和特殊字符。
```python
message = '''This is a
multi-line
string.'''
print(message)
```
### `\`（反斜杠转义字符）
用于对字符串中的特殊字符进行转义，将其视为普通字符。例如，`\'`表示单引号，`\"`表示双引号，`\\`表示反斜杠等。
```python
message = 'He said: "It\'s raining."'
print(message)  # 'He said: "It's raining."'
```
