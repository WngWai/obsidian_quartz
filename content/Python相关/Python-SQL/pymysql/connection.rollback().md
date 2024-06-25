在Python中，`connection.rollback()`函数用于回滚数据库事务，撤销之前的操作。

**函数定义**：
```python
connection.rollback()
```

**参数**：
`connection.rollback()`函数没有接受任何参数。

**示例**：
以下是使用`connection.rollback()`函数回滚数据库事务的示例：

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

    # 回滚事务
    conn.rollback()
except:
    # 如果发生错误，回滚事务
    conn.rollback()
finally:
    # 关闭游标和连接
    cursor.close()
    conn.close()
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例中，我们执行了一些数据库操作，如插入记录和更新记录。然后，通过调用`connection.rollback()`函数回滚事务，撤销之前的操作，使数据库回到事务开始之前的状态。

在示例中，我们在`try`块中执行了数据库操作。如果在执行期间发生了错误或异常，我们使用`except`块中的代码回滚事务，以确保数据库的一致性。

请注意，在执行完所有的数据库操作后，记得调用`cursor.close()`关闭游标对象，以及调用`conn.close()`关闭数据库连接，以释放资源。

以上是`connection.rollback()`函数的基本用法和示例。它是用于回滚数据库事务的函数，撤销之前的操作，使数据库回到事务开始之前的状态。