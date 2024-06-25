`pandas` 中的 `ExcelWriter` 对象可以让我们将 DataFrame 数据写入 Excel 文件。下面是一个简单的使用例子：

``` python
import pandas as pd

data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df = pd.DataFrame(data, columns=['Name', 'Age'])
with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)
```

上面的例子将一个 DataFrame 写入了一个名为 `output.xlsx` 的 Excel 文件的 `Sheet1` 工作表中。

下面是 `ExcelWriter` 对象的常用参数及功能：

- `path`：Excel 文件的路径，字符串类型。
- `mode`：打开 Excel 文件的方式，可选参数为 `'w'`（覆盖写入）和 `'a'`（追加写入），默认为 `'w'`。
- `engine`：写入 Excel 文件时使用的引擎，可选参数为`'openpyxl'`（默认值）和`'xlsxwriter'`。
- `datetime_format`：日期时间的格式，字符串类型，默认值为 `'yyyy-mm-dd hh:mm:ss'`。
- `date_format`：日期的格式，字符串类型，默认值为 `'yyyy-mm-dd'`。

创建 `ExcelWriter` 对象后，可以使用 `to_excel` 方法将 DataFrame 数据写入某个工作表中。同时，我们还可以使用 `writer.sheets['Sheet1']` 来访问已经写入的工作表，比如可以设置某个工作表的列宽或行高等属性。下面是一个包含这些操作的例子：

``` python
import pandas as pd

data = [['Alice', 25], ['Bob', 30], ['Charlie', 35], ['David',40]]
df = pd.DataFrame(data, columns=['Name', 'Age'])

# 创建 ExcelWriter 对象
with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    # 将 DataFrame 写入工作表 Sheet1 中
    df.to_excel(writer, sheet_name='Sheet1', index=False)

    # 修改 Sheet1 的列宽和行高
    worksheet = writer.sheets['Sheet1']
    worksheet.set_column('A:A', 15)
    worksheet.set_column('B:B', 15)
    worksheet.set_row(0, 30)  # 第 1 行的高度设置为 30
```

在上面的例子中，我们还通过 `set_column` 方法设置了 Sheet1 工作表的第一列和第二列的宽度为 15，通过 `set_row` 方法将第 1 行的高度设置为 30。

除了上面提到的 `to_excel` 方法外，`ExcelWriter` 对象还提供了以下两个方法：

- `save`：将写入的数据保存到 Excel 文件中。
- `close`：关闭 ExcelWriter 对象。

当我们写入了多个 DataFrame 数据时，如果需要将这些数据分别写入不同的工作表中，可以在创建 `ExcelWriter` 对象时指定 `sheet_name` 参数为 `None`，之后在调用 `to_excel` 方法时指定具体的工作表名。下面是一个包含这些操作的例子：

``` python
import pandas as pd

data1 = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df1 = pd.DataFrame(data1, columns=['Name', 'Age'])

data2 = [['David', 40], ['Eva', 27], ['Frank', 32]]
df2 = pd.DataFrame(data2, columns=['Name', 'Age'])

# 创建 ExcelWriter 对象
with pd.ExcelWriter('output.xlsx', mode='w') as writer:
    # 将 df1 写入 Sheet1 工作表中
    df1.to_excel(writer, sheet_name='Sheet1', index=False)

    # 将 df2 写入 Sheet2 工作表中
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

上面的例子中，我们将两个 DataFrame 分别写入了名为 `Sheet1` 和 `Sheet2` 的工作表中。