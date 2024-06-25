在Python中，`connection.commit()`函数用于提交数据库事务。

**函数定义**：
```python
connection.commit()
```

**参数**：
`connection.commit()`函数没有接受任何参数。

**示例**：
以下是使用`connection.commit()`函数提交数据库事务的示例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn
conn = pymysql.connect(host='localhost', user='username', password='password', database='mydb')
cursor = conn.cursor()

try:
    # 示例1：插入一条记录
    query = "INSERT INTO customers (name, email) VALUES ('John', 'john@example.com')"
    cursor.execute(query)

    # 示例2：更新记录
    query = "UPDATE customers SET email = 'new_email@example.com' WHERE id = 1"
    cursor.execute(query)

    # 提交事务
    conn.commit()
except:
    # 回滚事务
    conn.rollback()
finally:
    # 关闭游标和连接
    cursor.close()
    conn.close()
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例中，我们执行了一些数据库操作，如插入记录和更新记录。然后，通过调用`connection.commit()`函数提交事务，将之前的操作永久保存到数据库中。

如果在事务提交之前发生了错误或异常，我们可以通过调用`connection.rollback()`函数回滚事务，撤销之前的操作，以确保数据库的一致性。

请注意，在执行完所有的数据库操作后，记得调用`cursor.close()`关闭游标对象，以及调用`conn.close()`关闭数据库连接，以释放资源。

以上是`connection.commit()`函数的基本用法和示例。它是用于提交数据库事务的函数，将之前的操作永久保存到数据库中。