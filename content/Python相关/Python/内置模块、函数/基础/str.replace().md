是一个常用的字符串方法，用于替换字符串中的某些子串为另一个子串。下面是该函数的定义、参数介绍以及应用举例。
`replace()` 是Python字符串类（`str`）的一个方法。
```python
replace(old, new, count)
```

1. `old`（必需）：需要被替换的子串。
2. `new`（必需）：用于替换的新子串。
3. `count`（可选）：替换操作的最大次数。如果不指定或者为负数，则替换所有匹配的子串。

`replace()` 方法返回一个**新的字符串**，它是通过替换原始字符串中的子串而得到的。原始字符串不会被改变（因为字符串在Python中是不可变的）。


#### 例子1：基本替换

```python
python复制代码s = "Hello, world!"  new_s = s.replace("world", "Python")  print(new_s)  # 输出: Hello, Python!
```

#### 例子2：指定替换次数

```python
python复制代码s = "apple, apple, apple pie"  new_s = s.replace("apple", "orange", 2)  # 只替换前两个"apple"  print(new_s)  # 输出: orange, orange, apple pie
```

#### 例子3：不替换（`count` 设置为0）

虽然这看起来没有实际用途，但你可以将 `count` 设置为0来查看原始字符串而不进行任何替换。

```python
python复制代码s = "Hello, world!"  new_s = s.replace("world", "Python", 0)  # 不进行替换  print(new_s)  # 输出: Hello, world!
```

#### 例子4：替换不存在的子串

如果原始字符串中不存在需要被替换的子串，那么 `replace()` 方法将返回原始字符串的一个副本。

```python
python复制代码s = "Hello, world!"  new_s = s.replace("universe", "Python")  # "universe" 不在 s 中  print(new_s)  # 输出: Hello, world!
```