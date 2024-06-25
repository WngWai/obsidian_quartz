 是一个 Python 的内置函数，用于返回**指定路径下**的**所有文件和文件夹的列表**。这个函数返回一个包含目录内容的字符串列表。

指定位置下的所有**文件夹名称、文件名称字符串列表**

**定义：**
```python
import os

files_and_dirs = os.listdir(path)
```

**参数介绍：**
- `path`：要获取内容的目录路径。path为目录路径，默认为**当前目录**

**返回值：**
返回一个包含目录内容的字符串列表，列表中的每个元素都是目录中的一个文件或子目录的名称。

**举例：**
```python
import os

# 获取当前目录下的所有文件和子目录
current_directory = os.getcwd()
contents = os.listdir(current_directory)

# 打印结果
print("Contents of", current_directory, "are:")
for item in contents:
    print(item)
```

**输出：**
```
Contents of /path/to/current_directory are:
file1.txt
file2.txt
subdirectory1
subdirectory2
```

在这个示例中，`os.listdir()` 获取了当前目录下的所有文件和子目录，并将它们以列表形式返回。随后，我们遍历这个列表并打印每个文件和子目录的名称。