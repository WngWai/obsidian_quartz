在Python中，`cursor.fetchall()`函数用于从数据库中检索所有的记录。该函数返回一个**包含所有记录的列表**，每条记录都是一个**元组**。

**函数定义**：
```python
cursor.fetchall()
```


举例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn
conn = pymysql.connect(host='localhost', user='username', password='password', database='mydb')
cursor = conn.cursor()

# 示例：从customers表中检索所有记录
query = "SELECT * FROM customers"
cursor.execute(query)

rows = cursor.fetchall()
for row in rows:
    print(row)

# 关闭游标和连接
cursor.close()
conn.close()
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例中，我们执行了一个查询语句，使用`cursor.execute()`函数执行SQL查询。然后，通过调用`cursor.fetchall()`函数检索所有的记录。

`cursor.fetchall()`函数将返回一个包含所有记录的列表。然后，我们使用`for`循环遍历每条记录，并对其进行处理，如打印记录的值。

请注意，在完成所有的数据库操作后，记得调用`cursor.close()`关闭游标对象，以及调用`conn.close()`关闭数据库连接，以释放资源。

以上是`cursor.fetchall()`函数的基本用法和示例。它用于从数据库中检索所有的记录，并返回一个包含这些记录的列表。