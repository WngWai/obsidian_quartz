在 Python 中，**生成器对象**（Generator Object）是一种特殊的迭代器，用于逐步生成数据而不是一次性返回一个完整的数据集。|
生成器通过 `yield` 关键字定义，能够在函数执行过程中暂停并保存其状态，以便在下一次调用时继续执行。生成器通常用于处理大型数据集、懒加载数据和提高内存效率。

`yield` 是 Python 中的一个关键字，用于**定义生成器函数**。生成器函数返回一个**生成器对象**，可以迭代产生值，而不是一次性生成并返回所有值。**生成器函数中包含** `yield` 语句，当调用生成器的 `__next__()` 方法时，生成器函数执行到 `yield` 处停止，并将 `yield` 后面的值返回给调用方。下次调用 `__next__()` 方法时，生成器会从**上次停止的地方继续执行**，直到再次遇到 `yield` 或函数结束。

生成器对象具有**可迭代性**，直接next()就可以。

### 1. 生成器对象的定义

生成器对象通过生成器函数定义，生成器函数与普通函数类似，但使用 `yield` 关键字而不是 `return`。每次调用 `yield` 时，函数会暂停并返回一个值，但保留其执行状态，以便下一次迭代时继续从暂停的位置执行。

```python
def simple_generator():
    yield 1
    yield 2
    yield 3
```

使用生成器函数会返回一个生成器对象：

```python
gen = simple_generator()
print(type(gen))  # <class 'generator'>
```

### 2. 生成器对象的使用
生成器会在每次调用 `next()` 时暂停执行，并**从上次暂停的地方继续执行**，然后返回生成的值。当没有更多的值可以生成时，生成器会引发 `StopIteration` 异常。

生成器对象可以使用 `next()` 函数或 `for` 循环进行迭代：

```python
# 使用 next() 函数手动迭代
gen = simple_generator()
print(next(gen))  # 输出: 1
print(next(gen))  # 输出: 2
print(next(gen))  # 输出: 3
# print(next(gen))  # 再次调用 next() 会引发 StopIteration 异常

# 使用 for 循环迭代
for value in simple_generator():
    print(value)
    
# 输出: 1
# 输出: 2
# 输出: 3
```

### 3. 生成器的分类

- **简单生成器**：直接使用 `yield` 关键字生成值。
  ```python
  def count_up_to(max):
      count = 1
      while count <= max:
          yield count
          count += 1
  ```

- **生成器表达式**：使用**类似列表推导式**的语法定义生成器。
  ```python
  gen_expr = (x * x for x in range(5))
  for value in gen_expr:
      print(value)
  ```

- **无限生成器**：可以无限生成值直到手动停止（通常与条件和控制流结合使用）。
  ```python
  def infinite_sequence():
      num = 0
      while True:
          yield num
          num += 1
  ```

### 4. 生成器的应用举例

生成器在处理大型数据集、懒加载数据和流式处理等场景中非常有用。以下是几个实际应用的示例：

#### 4.1 读取大型文件
逐行读取大文件，避免一次性将整个文件加载到内存中。

```python
def read_large_file(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            yield line.strip()

for line in read_large_file('large_file.txt'):
    print(line)
```

#### 4.2 生成斐波那契数列
生成无限斐波那契数列。

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib_gen = fibonacci()
for _ in range(10):
    print(next(fib_gen))
```

#### 4.3 数据流处理
模拟数据流处理，例如生成传感器数据。

```python
import random
import time

def sensor_data_stream():
    while True:
        yield random.random()
        time.sleep(1)

for data in sensor_data_stream():
    print(data)
    # 添加终止条件以避免无限循环
    if data > 0.9:
        break
```

#### 4.4 分块处理大数据集
将大数据集分块处理，以避免内存耗尽。

```python
def chunked(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

data = range(100)
for chunk in chunked(data, 10):
    print(chunk)
```




### os.walk()返回的就是这种可迭代对象
