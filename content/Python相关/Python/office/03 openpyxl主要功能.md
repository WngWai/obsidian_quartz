python中的openpyxl包是一个用于读写Excel文件的库，它支持xlsx格式的文件，可以处理单元格数据、样式、格式、图表、图片等功能。

wb：workbook工作簿对象
ws：worksheet工作表对象 
### 工作簿（Workbook）
用于创建、打开、保存、复制、删除Excel文件，以及获取工作簿的属性和设置。
 
[[wb=openpyxl.Workbook()]]创建一个新的**工作簿对象**。
wb=load_workbook(filename)打开一个**已存在的Excel文件，返回一个工作簿对象**

[[wb.save(filename)]]**保存工作簿到指定的文件名**。
`copy_workbook(workbook)`：**复制**一个工作簿对象，返回一个新的工作簿对象。
`remove(workbook)`：**删除**一个工作簿对象。

属性：
[[sheet=wb.active]]**获取或设置**工作簿的当前**活动工作表**，初始创建工作簿对象时里面有张默认的工作表，不新建工作表，便会**激活默认**的
wb.sheetnames 获取工作簿中**所有工作表的名称列表**
wb.properties 获取或设置工作簿的**元数据**，如标题、作者、创建日期等。
wb.security 获取或设置工作簿的**安全选项**，如密码、只读、宏等。

### 工作表（Worksheet）
用于创建、删除、重命名、复制、移动、隐藏、冻结、筛选、排序等操作工作表，以及获取工作表的属性和设置。

[[sheet=create_sheet(title)]]在工作簿中**创建一个新的工作表**，可以指定标题和位置，返回一个**工作表对象**
sheet=`get_sheet_by_name(sheet_name)`：根据名称**获取工作簿中的工作表对象**，如果不存在则返回None。

[[wb.remove(ws)]]从工作簿中**删除一个工作表对象**。
ws=`copy_worksheet(ws)`**复制**一个工作表对象，返回一个新的工作表对象。
[[wb.move_sheet(ws, offset)]]将**工作表向前、后移动**，offset为正数表示向后移，为负数表示向前移。

属性：
ws.sheet_state`：获取或设置工作表的状态，如可见、隐藏、非激活等。
`ws.title`：获取或设置工作表的**标题**
[[ws.sheet_properties]]获取或设置工作表的**属性**，如标签颜色、缩放比例、方向等，得ws到工作表**属性字典**

**`ws.freeze_panes`**：获取或设置工作表的冻结窗格，可以冻结行、列或单元格。
`sheet.auto_filter`：获取或设置工作表的自动筛选区域，可以根据条件筛选数据。
`sheet.sort_state`：获取或设置工作表的排序状态，可以根据多个键值排序数据。

增：
[[ws.append()]]项ws中**添加一行数据**

### 单元格（Cell）
用于获取、设置、合并、拆分、插入、删除、格式化、公式等操作单元格，以及获取单元格的属性和设置。

`cell(row, column, value)`：获取或设置工作表中指定位置的单元格，可以指定值，返回一个单元格对象。
`merge_cells(range_string)`：合并工作表中指定范围的单元格，范围用字符串表示，如'A:B2'。
`unmerge_cells(range_string)`：拆分工作表中指定范围的单元格，范围用字符串表示，如'A:B2'。
`insert_rows(idx, amount)`：在工作表中指定位置插入指定数量的行，idx为行号，amount为数量。
`delete_rows(idx, amount)`：在工作表中指定位置删除指定数量的行，idx为行号，amount为数量。
`insert_cols(idx, amount)`：在工作表中指定位置插入指定数量的列，idx为列号，amount为数量。
`delete_cols(idx, amount)`：在工作表中指定位置删除指定数量的列，idx为列号，amount为数量。
`value`：获取或设置单元格的值，可以是字符串、数字、日期、时间、布尔值等。
`data_type`：获取或设置单元格的数据类型，如's'表示字符串，'n'表示数字，'d'表示日期等。
`number_format`：获取或设置单元格的数字格式，如'0.00%'表示百分比，'yyyy-mm-dd'表示日期等。
`style`：获取或设置单元格的样式，如字体、颜色、对齐、边框、填充等。
`formula`：获取或设置单元格的公式，如'=SUM(A:B2)'表示求和公式。

### 图表（Chart）
用于创建、添加、删除、修改、保存等操作图表，以及获取图表的属性和设置。

`BarChart()`：创建一个柱状图对象。
`LineChart()`：创建一个折线图对象。
`PieChart()`：创建一个饼图对象。
`ScatterChart()`：创建一个散点图对象。
`add_data(data, titles_from_data)`：向图表中添加数据，data为一个单元格范围，titles_from_data为是否从数据中获取标题。
`set_categories(labels)`：设置图表的类别标签，labels为一个单元格范围。
`title`：获取或设置图表的标题。
`style`：获取或设置图表的样式，如颜色、布局、图例等。
`x_axis`：获取或设置图表的x轴，如标题、刻度、网格线等。
`y_axis`：获取或设置图表的y轴，如标题、刻度、网格线等。
`add_chart(chart, anchor)`：向工作表中添加一个图表，chart为一个图表对象，anchor为一个单元格位置。
`save_chart(chart, filename)`：将图表保存为图片文件，chart为一个图表对象，filename为文件名。

