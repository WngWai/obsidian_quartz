`ExcelWriter`是`pandas`库中的一个类，用于**将一个或多个`DataFrame`对象保存到Excel文件中**。当使用`pandas`读写Excel文件时，可以使用`ExcelWriter`来对Excel表格进行**更加灵活**的操作，比如:

- 将多个`DataFrame`对象保存到Excel同一个文件的**多个或同一张工作表**中
- 在已有的Excel表格中**新增或覆盖**工作表
- 指定**输出数据框的位置和样式**，并进行**单元格或合并操作**

下面是`ExcelWriter`类的主要参数：

- `path`：指定Excel文件的**文件路径或handle**。如果文件不存在，则创建该文件。
- `engine`：指定使用的**Excel文件格式**，目前支持的格式包括：openpyxl、xlsxwriter、xlwt、pyxlsb、odf、numexpr。
 `openpyxl`：默认引擎，兼容性良好，支持 Excel 2010 及其更高版本的xlsx格式文件。
 `xlsxwriter`：速度较慢，但是功能很强大，支持写入图表。
 `xlwt`：支持 Excel 97-2003 文件格式的xls文件，但是不支持xlsx格式。
 `pyxlsb`：支持xlsb格式的二进制Excel文件，但是不支持xlsx格式。
 `odf`：支持Open Document Format for Office Applications的ods格式文件。
 `numexpr`：使用`numexpr`计算引擎来写入Excel文件，速度较快。

- `mode`：指定Excel文件的**打开模式**，支持的模式有：'w'，'a'，'r+'，'x'（仅当path指向的文件不存在时使用），默认为'w'。
- `date_format`：指定**日期格式**，默认值为yyyy-mm-dd
- `datetime_format`：指定**日期和时间格式**，默认值为yyyy-mm-dd hh:mm:ss。

在创建`ExcelWriter`对象后，可以使用它的`write()`方法将数据框写入Excel工作表中。`write()`方法的一些基本参数如下：

- `DataFrame`：指定需要写入Excel表格的数据框
- `sheet_name`：指定写入的工作表名
- `startrow`：指定输出内容的**起始行**，从0开始
- `startcol`：指定输出内容的**起始列**，从0开始
- `index`：是否输出数据框的**索引列**，默认为`True`
- `header`：是否输出数据框的**标题行**，默认为`True`

下面是一个用`ExcelWriter`将多个`DataFrame`保存到同一Excel文件不同工作表中的示例：

```python
import pandas as pd

# 创建数据框
data1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
data2 = pd.DataFrame({'c': [7, 8, 9], 'd': [10, 11, 12]})

# 创建ExcelWriter对象
writer = pd.ExcelWriter('example.xlsx')

# 将两个数据框写入同一Excel文件different sheet
data1.to_excel(writer, sheet_name='Sheet1')
data2.to_excel(writer, sheet_name='Sheet2')

# 保存Excel文件
writer.save()
```

在上面的示例中，我们使用ExcelWriter将两个数据框写入同一个在example.xlsx中的不同工作表中，每个数据框都作为独立的worksheet进行写入。