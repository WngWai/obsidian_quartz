在 Python 的 `os.path` 模块中，`os.path.split()` 函数用于拆分路径字符串为目录路径和文件名。以下是该函数的定义、参数和详细举例：

```python
os.path.split(path)
```

- **path** (str): 要拆分的路径字符串。


返回一个包含目录路径和文件名的元组。

### 详细举例：

```python
import os

# 定义一个文件路径
file_path = '/path/to/example.txt'

# 使用 os.path.split() 拆分路径
directory, filename = os.path.split(file_path)

# 打印结果
print(f"Directory: {directory}")
print(f"Filename: {filename}")
```

在这个例子中，我们定义了一个文件路径 `/path/to/example.txt`，然后使用 `os.path.split()` 函数拆分该路径。拆分后的结果是一个包含目录路径和文件名的元组，然后我们分别将其赋值给 `directory` 和 `filename` 变量。最后，我们打印了拆分后的目录和文件名。

这个例子中，输出结果将是：

```
Directory: /path/to
Filename: example.txt
```

`os.path.split()` 是一个常用的函数，用于从文件路径中提取目录和文件名信息。