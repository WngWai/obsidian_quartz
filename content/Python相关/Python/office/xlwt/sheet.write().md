在 `xlwt` 包中，`Sheet.write()` 方法用于在工作表中写入数据。以下是该方法的定义、参数和详细举例：

### `Sheet.write()` 方法介绍：

```python
Sheet.write(r, c, label='', style=Style.default_style)
```

### 参数：

- **r** (int): 行索引（从0开始）。
- **c** (int): 列索引（从0开始）。
- **label** (optional): 要写入的数据。
- **style** (optional): 单元格样式，默认为 `Style.default_style`。

### 详细举例：

下面是一个使用 `Sheet.write()` 方法的示例，演示如何创建一个 Excel 工作簿，添加工作表并写入数据：

```python
import xlwt

# 创建一个新的Excel工作簿
workbook = xlwt.Workbook()

# 添加一个工作表
sheet = workbook.add_sheet('Sheet1')

# 写入数据到工作表
sheet.write(0, 0, 'Name')
sheet.write(0, 1, 'Age')
sheet.write(1, 0, 'John')
sheet.write(1, 1, 25)
sheet.write(2, 0, 'Alice')
sheet.write(2, 1, 30)

# 保存工作簿到文件
workbook.save('example.xls')
```

在这个例子中，我们首先使用 `xlwt.Workbook()` 方法创建一个新的 Excel 工作簿对象。然后，使用 `add_sheet()` 方法向工作簿添加一个名为 'Sheet1' 的工作表。接着，使用 `write()` 方法在工作表中写入数据。

在 `write()` 方法的每个调用中，第一个参数是行索引（`r`），第二个参数是列索引（`c`），第三个参数是要写入的数据（`label`），可以是文本或数字等。在这个例子中，我们在第一行写入了标题，接着在第二行和第三行写入了姓名和年龄。

最后，使用 `save()` 方法保存工作簿到文件，这里保存为 'example.xls'。这是一个简单的示例，演示了如何使用 `Sheet.write()` 方法在 Excel 工作表中写入数据。