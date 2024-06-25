在Python中，`cursor.fetchone()`函数用于从数据库检索一条记录。

**函数定义**：
```python
cursor.fetchone()
```

**参数**：
`cursor.fetchone()`函数没有接受任何参数。

**返回值**：
该函数返回一个包含一条记录的元组或None（如果没有更多的记录可用）。

**示例**：
以下是使用`cursor.fetchone()`函数检索数据库中的记录的示例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn
conn = pymysql.connect(host='localhost', user='username', password='password', database='mydb')
cursor = conn.cursor()

# 示例：从customers表中检索一条记录
query = "SELECT * FROM customers WHERE id = 1"
cursor.execute(query)

row = cursor.fetchone()
if row is not None:
    print(row)
else:
    print("No records found.")

# 关闭游标和连接
cursor.close()
conn.close()
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例中，我们执行了一个查询语句，使用`cursor.execute()`函数执行SQL查询。然后，通过调用`cursor.fetchone()`函数从结果集中获取一条记录。

如果`cursor.fetchone()`函数返回的结果不为None，说明找到了一条记录，我们可以对该记录进行处理，如打印记录的值。如果返回的结果为None，说明没有更多的记录可用。

请注意，在完成所有的数据库操作后，记得调用`cursor.close()`关闭游标对象，以及调用`conn.close()`关闭数据库连接，以释放资源。

以上是`cursor.fetchone()`函数的基本用法和示例。它用于从数据库中检索一条记录，并返回该记录的元组或None。