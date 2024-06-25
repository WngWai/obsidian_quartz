可以使用`if`语句来筛选数据，并在数据为空值时跳过当前循环。以下是一个简单的例子，其中我们将使用一个名为`df`的DataFrame来存储数据，并根据`column_name`列的值进行筛选。如果数据为空值，我们将使用`continue`语句跳过当前循环。

```python
import pandas as pd

df = pd.DataFrame({"column_name": [1, 2, None, 4, None, 6]})  # 创建DataFrame

for index, row in df.iterrows():
    if pd.isna(row["column_name"]):  # 如果数据为空
        continue  # 跳过当前循环
    print(row["column_name"])  # 输出非空数据
```

在这个例子中，我们使用`pd.isna()`函数检查每一行的 "column_name" 列是否为空。如果为空，`continue`语句会跳过当前循环，不执行`print()`语句，而是直接进入下一次循环。否则，将会执行`print()`语句，输出非空数据。

需要注意的是，你可以根据实际情况来修改if语句来筛选你需要的数据，而`pd.isna()`函数是用于判断数据是否为空值的万能函数，可以适用于多种情况。