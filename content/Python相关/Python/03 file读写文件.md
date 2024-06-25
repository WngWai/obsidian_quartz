在Python中与文件操作相关的函数多数集中在内置的`open`函数以及返回的文件对象上。下面按功能分类介绍一些常见的文件操作函数和方法：

```python
import pandas as pd

# 检查是否已经导入数据
def check_data(path):
    try:
        # 尝试读取存储数据的文件
        with open('record.txt','r') as f:
            record = f.read().strip()
            if path in record:
                return False
            else:
                return True
    except:
        return True

# 将数据保存到文件中
def save_record(path):
    with open('record.txt','a') as f:
        f.write(path + '\n')

# 导入Excel数据到DataFrame对象
def import_data(path):
    if check_data(path):
        df = pd.read_excel(path)
        # 进行数据处理
        # ...
        # 保存导入数据的信息，避免重复导入
        save_record(path)
        return df
    else:
        print('数据已导入')
```

### 打开和关闭文件
[[open()]]  打开文件并返回相应的文件对象。不同的模式（`mode`参数）可以用于读取、写入或追加数据。
file.close()关闭一个已经打开的文件。关闭文件会自动释放系统资源。

[[with open...as上下文管理器]] 使用`with`语句和`open`函数来创建一个**上下文管理器**，这样可以保证文件用完之后会自动关闭。

### 文件属性
f.mode文件打开时使用的模式。
f.name文件的名字。
f.closed文件是否已被关闭。
f.encoding文件使用的编码。

### 文件操作
- 文件读取
	file.read(size=-1)读取文件中的数据。可选的`size`参数表示要读取的最大字节数。
	`file.readline(size=-1)`读取文件中的一行。可选的`size`参数表示要读取的最大字节数。
	`file.readlines(hint=-1)`读取文件中的所有行，并以列表形式返回。可选的`hint`参数表示要读取的大概字节数。

- 文件写入和追加
	file.write(string)将`string`写入文件。返回写入的字符数。
	`file.writelines(lines)`将一个字符串列表`lines`写入文件。不会自动添加换行符。
	`file.flush()` 刷新文件内部缓冲区，将缓冲区内容写入到文件中。
