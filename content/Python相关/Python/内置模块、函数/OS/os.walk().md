os.walk()返回一**生成器**，**在for in 迭代中**，每次得到**某一层**的root(当前位置路径)、dirs(当前位置下有哪些文件夹，返回的是**文件夹名列表**)、files（当前位置下有哪些文件，返回的是**文件名列表**）

是 Python 中用于**遍历文件夹**中所有文件和子文件夹的函数。它提供了一种便捷的方式来**递归地访问**文件系统中的文件和文件夹。

```python
for root, dirs, files in os.walk(top, topdown=True, onerror=None, followlinks=False):
```


![[Pasted image 20240120151629.png]]
遍历第一层
root：先遍历top；
dirs: 当前top路径下的文件**夹**列表，如[dirname1, dirname2...]；
files: 当前top路径下的文件列表，如[filename1.txt, filename2.txt, ...]；
遍历第二层
root：为dirname1；
dirs: 当前dirname1路径下的文件夹列表，如[dirname1/1, dirname1/2...]；
files: 当前dirname1路径下的文件列表，如[filename3, filename4...]；

root：为dirname2；
dirs: 当前dirname2路径下的文件夹列表，如[dirname2/1, dirname2/2...]；
files: 当前dirname2路径下的文件列表，如[filename5, filename6...]；

如此逐层遍历，最后返回的就是根文件夹下的**所有文件夹和文件信息**！
top为根目录路径，
topdown为是否自顶向下遍历，默认为True，
onerror为遇到错误时的回调函数，默认为None，
followlinks为是否跟随符号链接，默认为False。

- `top`：要遍历的**根文件夹路径**，就是输入的文件路径。**子文件**指得是根文件下的子文件。先遍历的**根目录路径**，就是最上面的节点，下面的叫**叶节点、叶目录**
- `topdown`（可选）：树形图遍历。如果为 True，则首**先遍历根文件夹**，然后遍历**子文件夹**；如果为 False，则首先遍历子文件夹，然后遍历根文件夹。默认值为 **True**。
- `onerror`（可选）：当访问文件或文件夹时**出现错误时**的处理函数。默认值为**None**。
- `followlinks`（可选）：如果为 False，则**忽略符号链接**。默认值为 False。

```python
import os

# 遍历文件夹并打印所有文件路径
folder_path = "path/to/folder"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
```

输出结果：
```
path/to/folder/file1.txt
path/to/folder/subfolder1/file2.txt
path/to/folder/subfolder1/file3.txt
path/to/folder/subfolder2/file4.txt
...
```

在上述示例中，我们使用了 `os.walk()` 函数遍历文件夹及其子文件夹，并打印所有文件的路径。

首先，我们指定了要遍历的根文件夹路径 `folder_path`。
然后，使用 `for root, dirs, files in os.walk(folder_path)` 进行遍历。在每次迭代中，我们可以对当前文件夹的处理逻辑进行操作。

通过遍历文件夹中的每个子文件夹和文件，我们可以使用 `os.path.join()` 函数来**构建完整的文件路径**，并对每个文件路径进行操作，如打印路径。

需要注意的是，`os.walk()` 函数会递归地遍历文件夹及其子文件夹，因此可能需要注意**处理大量文件时的性能问题**，以及**对符号链接**的处理方式（通过 `followlinks` 参数控制）。


在 Python 中，`os.walk()` 是 `os` 模块中的一个函数，用于遍历指定目录及其子目录下的所有文件和子目录。这个函数**返回一个生成器**，**每次迭代**产生一个包含当前目录路径、子目录列表和文件列表的元组。

以下是 `os.walk()` 函数的基本信息：

**功能：** **遍历指定目录及其子目录下的所有文件和子目录**。

**定义：**
```python
import os

for root, dirs, files in os.walk(top, topdown=True, onerror=None, followlinks=False):
```

**举例：**
```python
import os

# 遍历当前目录及其子目录下的所有文件和子目录
for root, dirs, files in os.walk(".", topdown=True):
    print(f"Current Directory: {root}")
    print(f"Subdirectories: {dirs}")
    print(f"Files: {files}")
    print("=" * 50)
```

**输出：**
```
Current Directory: .
Subdirectories: ['subdir1', 'subdir2']
Files: ['file1.txt', 'file2.txt']
==================================================
Current Directory: .\subdir1
Subdirectories: ['subsubdir']
Files: ['file3.txt']
==================================================
Current Directory: .\subdir1\subsubdir
Subdirectories: []
Files: ['file4.txt']
==================================================
Current Directory: .\subdir2
Subdirectories: []
Files: ['file5.txt']
==================================================
```
