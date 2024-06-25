`xlwt` 是一个用于操作 Excel 文件的 Python 模块，它支持创建和修改 Excel 文件，并提供了一些常用的样式设置功能。以下是 `xlwt` 模块中一些主要的函数和功能，按功能分类介绍：

xlwt只能写入**xls**格式的文件，不支持写入xlsx文件。它的写入**效率较高，但功能较单一**，不能设置单元格格式或插入图片等。

openpyxl支持读写xlsx文件，功能较广泛，可以设置单元格格式、插入图片、表格、图表等。它的**读写效率较低**，但可以通过开启read_only和write_only模式来提升性能。

xlsxwriter用于创建xlsx文件，支持图片、表格、图表、格式等，功能与openpyxl相似，优点是还支持VBA文件导入，缺点是**只能创建新的Excel文件，不能读取或修改已经存在的Excel文件**。

```python
import xlwt

# 创建工作簿
workbook = xlwt.Workbook()

# 创建工作表
sheet = workbook.add_sheet('Sheet1')

# 写入数据
sheet.write(0, 0, 'Hello')
sheet.write(0, 1, 'World')

# 合并单元格
sheet.merge(2, 2, 2, 4)


# 设置样式
style = xlwt.Style.easyxf('font: bold 1, color red;')
sheet.write(1, 0, 'Styled', style)
sheet.col(0).width = 5000
sheet.row(0).height = 500

# 设置边框和背景色
border_style = xlwt.Style.easyxf('borders: left double, right double, top double, bottom double;')
pattern_style = xlwt.Style.easyxf('pattern: pattern solid, fore_color yellow;')
sheet.write(3, 0, 'Bordered and Yellow Background', xlwt.Style.XFStyle(border_style, pattern_style))

# 保存工作簿到文件
workbook.save('example.xls')
```


### 1.  建簿、表，保存
创建 Excel 文件和工作表：
[[wb=xlwt.Workbook()]]创建一个 **Excel 工作簿对象**。
[[sheet=wb.add_sheet()]]在工作簿中添加一个**工作表**

保存 Excel 文件
wb.save将工作簿对象保存为xlx文件
### 2. 表中数据处理
[[sheet.write()]]在工作表中写入数据到指定单元格
sheet.merge()合并单元格，行**跨行和跨列数据的合并**

### 3.样式设置
设置列宽和行高
Sheet.col设置列的宽度。
Sheet.row设置行的高度。

设置单元格内容和样式、背景色、日期和数字格式：
如没必要，手动调整吧？
**`Style.easyxf` 方法：** 创建一个样式对象。
 **`Style.Font` 类：** 创建字体样式对象。
 **`Style.XFStyle` 类：** 创建单元格样式对象。
 **`Style.Borders` 类：** 创建边框样式对象。
 **`Style.Pattern` 类：** 创建图案样式对象，用于设置背景色。
**`Style.num_format_str` 属性：** 设置数字格式。

