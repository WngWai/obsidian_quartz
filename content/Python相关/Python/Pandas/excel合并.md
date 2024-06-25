在Python中，可以使用`pandas`库来处理Excel文件。`pandas`提供了`read_excel()`函数来读取Excel文件，可以使用`concat()`函数合并多个Excel文件中的表格内容。

```python
import pandas as pd

# 读取第一个表格
df1 = pd.read_excel('file1.xlsx', sheet_name='Sheet1')

# 读取第二个表格
df2 = pd.read_excel('file2.xlsx', sheet_name='Sheet1')

# 将两个表格内容合并
merged_data = pd.concat([df1, df2])

# 将合并结果写入新文件
writer = pd.ExcelWriter('merged_data.xlsx')
merged_data.to_excel(writer, index=False, sheet_name='Sheet1')
writer.save()
```

在这个示例中，我们首先使用`pd.read_excel`函数从文件中读取两个表格。然后使用`pd.concat`函数将这两个表格合并。`pd.concat`函数的第一个参数是一个包含要合并的不同数据帧（Excel表格）的列表。最后，将合并后的结果写入到一个新的Excel文件中。