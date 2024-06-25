在Python中，`cursor.description`属性用于获取查询结果的元数据，包括每个查询结果列的名称、类型和其他属性。

**属性定义**：
```python
cursor.description
```

**参数**：
`cursor.description`属性没有接受任何参数。

**返回值**：
该属性返回一个描述查询结果的元组列表。每个元组包含以下信息：
- `name`：列的名称。
- `type_code`：列的数据类型代码。这是一个整数值，可以使用数据库模块的类型常量进行解释。
- `display_size`：在结果集中显示该列所需的宽度，以字符为单位。
- `internal_size`：列的内部存储大小，以字节为单位。对于可变长度数据类型，此值为None。
- `precision`：列的精度（总位数）。
- `scale`：列的小数部分的位数。
- `null_ok`：表示该列是否允许为空。如果允许为空，值为True；否则，值为False。

**示例**：
以下是使用`cursor.description`属性获取查询结果的元数据的示例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn
conn = pymysql.connect(host='localhost', user='username', password='password', database='mydb')
cursor = conn.cursor()

# 示例：执行查询并获取元数据
query = "SELECT * FROM customers"
cursor.execute(query)

# 获取查询结果的元数据
metadata = cursor.description
for column in metadata:
    name = column[0]
    type_code = column[1]
    display_size = column[2]
    internal_size = column[3]
    precision = column[4]
    scale = column[5]
    null_ok = column[6]

    print("Column Name:", name)
    print("Type Code:", type_code)
    print("Display Size:", display_size)
    print("Internal Size:", internal_size)
    print("Precision:", precision)
    print("Scale:", scale)
    print("Nullable:", null_ok)
    print()

# 关闭游标和连接
cursor.close()
conn.close()
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例中，我们执行了一个查询语句，使用`cursor.execute()`函数执行SQL查询。

然后，通过访问`cursor.description`属性，我们获取了查询结果的元数据。我们使用一个`for`循环遍历每个列的元数据，并打印出各个属性的值，如列名、数据类型、显示大小等。

请注意，在执行查询之前，必须先调用`cursor.execute()`函数执行查询语句，以确保`cursor.description`属性能够获取到正确的元数据。

以上是`cursor.description`属性的基本用法和示例。它用于获取查询结果的元数据，包括每个查询结果列的名称、类型和其他属性。