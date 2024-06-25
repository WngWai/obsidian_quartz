在 `openpyxl` 包中，`create_sheet()` 函数用于创建一个新的工作表（Sheet）。以下是该函数的定义、参数和详细举例：

```python
create_sheet(title=None, index=None)
```

- **title** (str, optional): 新工作表的**标题**，如果不提供则生成**默认标题**。
- **index** (int, optional): 新工作表的**索引位置**，如果不提供则**添加到最后**。

### 返回值：

返回一个新创建的 `Worksheet` 对象，即新的工作表。

### 详细举例：

```python
from openpyxl import Workbook

# 创建一个新的工作簿
workbook = Workbook()

# 创建一个新的工作表
new_sheet = workbook.create_sheet(title="MySheet", index=0)

# 在新工作表中写入数据
new_sheet['A1'] = 'Hello'
new_sheet['B1'] = 'World'

# 保存工作簿到文件
workbook.save('example.xlsx')
```

在这个例子中，我们首先创建一个新的工作簿对象，然后使用 `create_sheet()` 方法创建一个新的工作表，同时指定了工作表的标题为 "MySheet"，并将它插入到索引位置 0。接着，在新工作表中写入一些数据。最后，使用 `save()` 方法将工作簿保存到名为 'example.xlsx' 的文件中。

如果不提供 `title` 参数，则工作表将使用默认标题（Sheet1, Sheet2, 等）。如果不提供 `index` 参数，则新工作表将被添加到工作簿的最后。工作表的索引位置从 0 开始，表示第一个工作表。

这个函数提供了一种方便的方式来在工作簿中创建新的工作表，并可以指定标题和索引位置。