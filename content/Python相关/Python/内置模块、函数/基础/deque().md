在 Python 中, `deque()` 函数来自于 `collections` 模块,它用于创建双端队列(Deque,全称为"Double-Ended Queue")。

```python
collections.deque([iterable[, maxlen]])
```


**参数介绍**:
- `iterable`: 可选参数,用于初始化队列的元素。
- `maxlen`: 可选参数,用于设置队列的最大长度。如果省略,则队列大小不受限制。

**返回值**:
返回一个新的双端队列对象。

**应用举例**:

1. **基本使用**:
```python
from collections import deque

# 创建一个空的双端队列
queue = deque()

# 添加元素到队列末尾
queue.append(1)
queue.append(2)
queue.append(3)
print(queue)  # 输出: deque([1, 2, 3])

# 从队列头部删除元素
print(queue.popleft())  # 输出: 1
print(queue)  # 输出: deque([2, 3])
```

2. **设置最大长度**:
```python
from collections import deque

# 创建一个最大长度为 3 的双端队列
queue = deque(maxlen=3)

# 添加元素
queue.append(1)
queue.append(2)
queue.append(3)
print(queue)  # 输出: deque([1, 2, 3])

# 继续添加元素,超出最大长度后,队列头部的元素会自动被删除
queue.append(4)
print(queue)  # 输出: deque([2, 3, 4])
```

3. **双端操作**:
```python
from collections import deque

queue = deque([1, 2, 3])

# 从队列左端添加元素
queue.appendleft(0)
print(queue)  # 输出: deque([0, 1, 2, 3])

# 从队列右端删除元素
print(queue.pop())  # 输出: 3
print(queue)  # 输出: deque([0, 1, 2])

# 从队列左端删除元素
print(queue.popleft())  # 输出: 0
print(queue)  # 输出: deque([1, 2])
```

4. **迭代双端队列**:
```python
from collections import deque

queue = deque([1, 2, 3, 4, 5])

# 迭代队列元素
for item in queue:
    print(item, end=" ")  # 输出: 1 2 3 4 5

print()

# 反向迭代队列元素
for item in reversed(queue):
    print(item, end=" ")  # 输出: 5 4 3 2 1
```

总的来说, `deque()` 函数提供了一个具有双端特性的队列数据结构,在需要快速地在队列两端添加或删除元素的场景下非常有用,例如广度优先搜索、滑动窗口等算法。