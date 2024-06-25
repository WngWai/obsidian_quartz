在 `openpyxl` 包中，`active` 不是一个函数，而是 `Workbook` 对象的一个属性，用于获取或设置当前活动的工作表（即当前选中的工作表）。以下是该属性的介绍和详细示例：


```python
workbook.active
```

返回当前活动的工作表对象

### 详细举例：

```python
from openpyxl import Workbook

# 创建一个新的工作簿
workbook = Workbook()

# 获取当前活动的工作表
active_sheet = workbook.active

# 在活动的工作表中写入数据
active_sheet['A1'] = 'Hello'
active_sheet['B1'] = 'World'

# 保存工作簿到文件
workbook.save('example.xlsx')
```

在这个例子中，我们创建一个新的工作簿对象，并通过 `active` 属性获取当前活动的工作表（默认是第一个工作表）。然后，在活动的工作表中写入一些数据。最后，使用 `save()` 方法将工作簿保存到名为 'example.xlsx' 的文件中。

如果需要切换到其他工作表，可以使用 `workbook.active = sheet_number` 来设置新的活动工作表，其中 `sheet_number` 是工作表的索引。索引从 1 开始，表示第一个工作表。例如，`workbook.active = 2` 将把第二个工作表设置为活动工作表。

请注意，`openpyxl` 中的索引是从 1 开始的，而不是从 0 开始。