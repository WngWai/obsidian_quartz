在 Python 的 `os` 模块中，`os.mkdir()` 函数用于创建一个新的目录。以下是该函数的定义、参数和详细举例：

```python
import os
os.mkdir('/home/user/test', 0o755)
```


```python
os.mkdir(path, mode=0o777, *, dir_fd=None)
```

- **path** (str): 要创建的目录的路径。
- **mode** (optional): 目录的权限模式，默认为 `0o777`。在大多数系统上，该参数可以省略。
- **dir_fd** (optional): 如果给定，表示**文件描述符**，用于指定在哪个文件描述符上执行操作。

### 详细举例：

```python
import os

# 定义目录路径
new_directory = 'example_directory'

try:
    # 使用 os.mkdir() 创建新目录
    os.mkdir(new_directory)
    print(f"Directory '{new_directory}' created successfully.")
except OSError as e:
    print(f"Failed to create directory '{new_directory}': {e}")
```

在这个例子中，我们使用 `os.mkdir()` 函数创建一个名为 'example_directory' 的新目录。首先定义了目录路径 `new_directory`，然后在 `try` 块中使用 `os.mkdir(new_directory)` 创建目录。如果创建成功，将输出成功的消息，否则将捕获 `OSError` 异常，输出相应的错误信息。

需要注意的是，如果指定的目录已经存在，`os.mkdir()` 将引发 `OSError`。如果你想在目录已经存在时不引发异常，可以使用 `os.makedirs()` 函数。该函数可以递归创建目录，如果目录已经存在，则不引发异常。


- 创建一个名为test的目录，路径为当前目录：

```python
import os
os.mkdir('test')
```

- 创建一个名为test的目录，路径为D盘根目录：

```python
import os
os.mkdir('D:\\test')
```

- 创建一个名为test的目录，路径为/home/user/，并指定权限为0o755（八进制）：

```python
import os
os.mkdir('/home/user/test', 0o755)
```

