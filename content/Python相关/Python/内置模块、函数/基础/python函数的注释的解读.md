在Python中，类型提示是用于指示函数参数或返回值的预期类型的一种方式。它们**为开发者提供了额外的信息，以帮助确保函数被正确地使用**。

在给出的代码片段中：

```python
def read_lines_from_file(path: str) -> Iterator[str]:
```

1. `path: str`：这是一个类型提示，它指示`path`参数应该是一个字符串（`str`）。
2. `-> Iterator[str]`：这也是一个类型提示，它指示该函数将返回一个迭代器（`Iterator`），该迭代器产生字符串（`str`）。

- `path: str`：这告诉调用者，他们应该传递一个字符串作为`path`参数。
- `-> Iterator[str]`：这告诉调用者，该函数将返回一个迭代器，该迭代器可以用来遍历字符串。


在Python中，当你看到一个函数被定义为返回 `-> None`，这意味着该函数不返回任何值。`None` 是Python中的一个特殊对象，表示空或无。

例如：
```python
def say_hello(name):      
	print(f"Hello, {name}")    
	return None
```

在上面的例子中，`say_hello` 函数打印一个问候语，但并不返回任何有用的值，所以它返回 `None`。如果你调用这个函数并试图获取其返回值，你会得到 `None`。例如：

```python
result = say_hello("Alice") 
	print(result)  # 输出: None
```

### 常用的数据类型
```python
from datetime import datetime

int # 整型
str # 字符串
float # 浮点型
bool # 布尔值
datetime # 时间
```

### 其他例子
```python
def drop_duplicates(
	self,
	subset: Hashable | Sequence[Hashable] | None = None,
	*,
	keep: Literal["first", "last", False] = "first",
	inplace: bool = False,
	ignore_index: bool = False) -> Any
```
1. **self**：
   - 这是方法的对象实例，即调用该方法的DataFrame对象。
   2.**subset**：
   - 这是一个可选参数，用于指定哪些列的值应该被用来判断重复行。
   - 它可以是以下几种类型之一：
     - `Hashable`：**任何可哈希的对象**，如整数、浮点数、字符串、元组等。
     - `Sequence[Hashable]`：一个可哈希对象的序列，如列表、元组等。
     - `None`：不指定特定的列，而是**使用DataFrame的所有列**。
   - 如果指定了`subset`，只有这些列的值会被用来判断重复行。
   你可能会指定一列或多列作为subset参数，这些列的值必须是可哈希的，因为Pandas需要使用这些值来确定哪些行是重复的。如果这些值是不可哈希的（例如，是列表或字典），那么Pandas将无法确定这些行的唯一性，因此无法正确地删除重复行。
1. **keep**：
   - 这是一个关键字参数，用于指定如何处理重复行。
   - 它可以是以下值之一：
     - `"first"`：保留第一次出现的重复行，删除后续出现的重复行。
     - `"last"`：保留最后一次出现的重复行，删除之前的重复行。
     - `False`：删除所有重复行，只保留第一个出现的行。
   - 默认值是`"first"`。
2. **inplace**：
   - 这是一个关键字参数，用于指定是否在原地修改DataFrame，而不是返回一个新的DataFrame。
   - 如果为`True`，函数将在原地修改DataFrame，不返回任何值。
   - 如果为`False`（默认值），函数将返回一个新的DataFrame，不修改原始DataFrame。
3. **ignore_index**：
   - 这是一个关键字参数，用于指定是否在删除重复行后重置索引。
   - 如果为`True`，函数将在删除重复行后重置DataFrame的索引，并返回一个新的索引。
   - 如果为`False`（默认值），DataFrame的索引将保持不变。
返回值（**-> Any**）：
   - 函数返回一个任意类型的对象，通常是新的DataFrame，其中重复行已被删除。
