在 `openpyxl` 包中，`remove()` 方法用于从 `Workbook` 对象中移除指定的工作表。以下是该方法的定义、参数和详细举例：

```python
Workbook.remove(sheet)
```

- **sheet** (`Worksheet`): 要移除的工作表对象。

### 返回值：

该方法没有返回值。

### 详细举例：

```python
from openpyxl import Workbook

# 创建一个新的工作簿
workbook = Workbook()

# 创建两个工作表
sheet1 = workbook.active
sheet2 = workbook.create_sheet(title="Sheet2")

# 在工作表中写入数据
sheet1['A1'] = 'Sheet 1 Data'
sheet2['A1'] = 'Sheet 2 Data'

# 移除第一个工作表
workbook.remove(sheet1)

# 保存工作簿到文件
workbook.save('example.xlsx')
```

在这个例子中，我们首先创建一个新的工作簿对象，并默认创建了一个工作表（`sheet1`）和一个带有指定标题的工作表（`sheet2`）。然后，在两个工作表中写入一些数据。最后，我们使用 `remove()` 方法移除了第一个工作表（`sheet1`）。

在调用 `remove()` 方法后，`sheet1` 被从工作簿中移除，不再存在于工作簿中。工作表的内容和样式都被清除。这个方法允许你根据需要动态地添加或移除工作表。