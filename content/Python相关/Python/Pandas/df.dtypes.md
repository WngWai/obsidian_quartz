在pandas库中，`df.dtypes`是一个**属性**。它用于获取数据框中每个列的数据类型。
**属性定义**：
```python
df.dtypes
```

**参数**：
此属性没有参数。

下面是一个示例，演示如何使用`df.dtypes`属性获取数据框中每列的数据类型：

```python
import pandas as pd

# 准备数据
data = {'Name': ['John', 'Mike', 'Sarah'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}

df = pd.DataFrame(data)

# 获取每列的数据类型
data_types = df.dtypes

print(data_types)
```

输出：
```
Name    object
Age      int64
City    object
dtype: object
```

在上面的示例中，我们首先创建了一个数据字典`data`，其中包含三个列：`Name`、`Age`和`City`。然后，我们使用`pd.DataFrame()`函数将数据字典转换为数据框`df`。接下来，我们使用`df.dtypes`属性获取每列的数据类型，并将结果存储在`data_types`变量中。

输出结果显示了每列的名称和相应的数据类型。在示例中，`Name`和`City`列的数据类型是`object`，表示字符串类型，而`Age`列的数据类型是`int64`，表示整数类型。

通过使用`df.dtypes`属性，您可以方便地查看数据框中每列的数据类型。这对于数据的类型检查和处理非常有用，例如数据类型转换、缺失值处理等。

请注意，`df.dtypes`属性返回的结果是一个`Series`对象，其中索引是列名，值是数据类型。您可以通过`data_types['ColumnName']`的方式访问特定列的数据类型。

希望这个解答对您有帮助！