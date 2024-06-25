在Python中，`connection.cursor()`函数用于**创建一个游标对象**，该游标对象用于执行SQL查询和操作数据库。

**函数定义**：
```python
connection.cursor(cursor=None)
```

**参数**：
以下是`connection.cursor()`函数中的参数：

- `cursor`：游标类型。默认为`None`，表示使用**默认的游标**类型。

**示例**：
以下是使用`connection.cursor()`函数创建游标对象的示例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn

# 示例1：使用默认游标
cursor1 = conn.cursor()

# 示例2：使用指定游标类型
cursor2 = conn.cursor(pymysql.cursors.DictCursor)
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`。

在示例1中，我们使用`connection.cursor()`函数创建一个游标对象`cursor1`，该游标对象使用默认的游标类型。

在示例2中，我们使用`connection.cursor()`函数创建一个游标对象`cursor2`，该游标对象使用`pymysql.cursors.DictCursor`类型的游标。`DictCursor`类型的游标会返回字典形式的结果，方便通过列名访问查询结果的数据。

创建游标对象后，我们可以使用游标对象执行SQL查询和操作数据库，例如执行`execute()`函数、`fetchall()`函数等。

请注意，在执行完SQL查询和操作后，记得通过`close()`函数关闭游标对象和数据库连接，以释放资源。

以上是`connection.cursor()`函数的基本用法和示例。该函数在Python中连接数据库后，常用于创建游标对象以便进行数据库操作。