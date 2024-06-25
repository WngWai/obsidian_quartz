在Python中，`cursor.fetchmany()`函数用于从数据库中检索多条记录。

**函数定义**：
```python
cursor.fetchmany(size=cursor.arraysize)
```

**参数**：
`cursor.fetchmany()`函数接受一个可选的参数`size`，表示要检索的记录的数量。默认情况下，`size`的值是`cursor.arraysize`，它是游标对象的属性，表示每次从数据库中检索的默认记录数量。

**返回值**：
该函数返回一个包含多条记录的列表。每条记录都是一个元组。

**示例**：
以下是使用`cursor.fetchmany()`函数检索数据库中多条记录的示例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn
conn = pymysql.connect(host='localhost', user='username', password='password', database='mydb')
cursor = conn.cursor()

# 示例：从customers表中检索多条记录
query = "SELECT * FROM customers"
cursor.execute(query)

rows = cursor.fetchmany(size=3)
for row in rows:
    print(row)

# 关闭游标和连接
cursor.close()
conn.close()
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例中，我们执行了一个查询语句，使用`cursor.execute()`函数执行SQL查询。然后，通过调用`cursor.fetchmany()`函数检索多条记录。

在示例中，我们通过`size=3`指定了要检索的记录数量为3。`cursor.fetchmany()`函数将返回一个包含3条记录的列表。然后，我们使用`for`循环遍历每条记录，并对其进行处理，如打印记录的值。

请注意，在完成所有的数据库操作后，记得调用`cursor.close()`关闭游标对象，以及调用`conn.close()`关闭数据库连接，以释放资源。

以上是`cursor.fetchmany()`函数的基本用法和示例。它用于从数据库中检索多条记录，并返回一个包含这些记录的列表。可以通过可选参数`size`指定要检索的记录数量。