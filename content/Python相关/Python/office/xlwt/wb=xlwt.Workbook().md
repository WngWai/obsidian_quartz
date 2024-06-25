在 Python 的 `xlwt` 包中，`xlwt.Workbook()` 方法用于创建一个新的 Excel 工作簿对象。以下是该方法的定义、参数和详细举例：

### `xlwt.Workbook()` 方法介绍：

```python
xlwt.Workbook(encoding='utf-8', style_compression=0)
```

### 参数：

- **encoding** (optional): Excel 文件的编码方式，默认为 'utf-8'。
- **style_compression** (optional): 样式压缩级别，**默认为 0**。通常情况下，可以使用默认值。

### 详细举例：

首先，确保已经安装了 `xlwt` 包。可以使用以下命令安装：

```bash
pip install xlwt
```

然后，我们可以使用 `xlwt.Workbook()` 方法创建一个 Excel 工作簿，并向其中添加工作表和数据：

```python
import xlwt

# 创建一个新的Excel工作簿
workbook = xlwt.Workbook()

# 添加一个工作表
sheet = workbook.add_sheet('Sheet1')

# 在工作表中写入数据
sheet.write(0, 0, 'Name')
sheet.write(0, 1, 'Age')
sheet.write(1, 0, 'John')
sheet.write(1, 1, 25)
sheet.write(2, 0, 'Alice')
sheet.write(2, 1, 30)

# 保存工作簿到文件
workbook.save('example.xls')
```

在这个例子中，我们首先使用 `xlwt.Workbook()` 方法创建一个新的 Excel 工作簿对象。然后，使用 `add_sheet()` 方法向工作簿添加一个名为 'Sheet1' 的工作表。接着，使用 `write()` 方法在工作表中写入数据。最后，使用 `save()` 方法保存工作簿到文件，这里保存为 'example.xls'。

这是一个简单的例子，演示了如何使用 `xlwt` 创建 Excel 工作簿、添加工作表和写入数据。根据实际需求，可以使用更多的 `xlwt` 方法来设置样式、合并单元格等。