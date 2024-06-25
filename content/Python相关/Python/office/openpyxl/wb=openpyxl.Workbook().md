在 `openpyxl` 包中，`Workbook()` 函数用于创建一个新的 Excel 工作簿对象。以下是该函数的定义、参数和详细举例：


```python
openpyxl.Workbook()
```

该函数没有接受任何参数。
返回一个新的 `Workbook` 对象，即一个空的 Excel 工作簿。

### 详细举例：

```python
from openpyxl import Workbook

# 创建一个新的工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 在工作表中写入数据
sheet['A1'] = 'Hello'
sheet['B1'] = 'World'

# 保存工作簿到文件
workbook.save('example.xlsx')
```

在这个例子中，我们使用 `Workbook()` 函数创建一个新的工作簿对象。然后，通过 `active` 属性获取默认的工作表，并在工作表中写入一些数据。最后，使用 `save()` 方法将工作簿保存到名为 'example.xlsx' 的文件中。

创建工作簿后，可以通过不同的方法和属性来操作工作簿，例如添加新的工作表、获取工作表、修改单元格内容等。上述示例中的简单用法是创建一个包含默认工作表的工作簿，并在其中写入数据。