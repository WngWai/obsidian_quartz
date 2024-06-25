在Python中，`cursor.execute()`函数用于执行SQL查询或操作数据库的命令。

**函数定义**：
```python
cursor.execute(operation, params=None, multi=False)
```

**参数**：
以下是`cursor.execute()`函数中的参数：

- `operation`：要执行的SQL查询或操作数据库的命令。它可以是包含SQL语句的字符串，也可以是使用参数占位符的字符串。

- `params`：可选参数，用于传递SQL查询中的参数值。它可以是单个值、元组或字典。

- `multi`：可选参数，用于指示是否执行多个SQL语句。默认为`False`，表示只执行单个SQL语句。

**示例**：
以下是使用`cursor.execute()`函数执行SQL查询和操作数据库的示例：

```python
import pymysql

# 假设已经建立了与数据库的连接 conn
cursor = conn.cursor()

# 示例1：执行SQL查询
query = "SELECT * FROM customers"
cursor.execute(query)

# 示例2：执行带参数的SQL查询
query = "SELECT * FROM customers WHERE id = %s"
params = (1,)
cursor.execute(query, params)

# 示例3：执行多个SQL语句
query1 = "INSERT INTO customers (name, email) VALUES (%s, %s)"
query2 = "SELECT * FROM customers"
params1 = ("John", "john@example.com")
cursor.execute(query1, params1, multi=True)
cursor.execute(query2)

# 示例4：使用字典方式传递参数
query = "SELECT * FROM customers WHERE id = %(id)s"
params = {"id": 1}
cursor.execute(query, params)
```

在上述示例中，我们首先导入了`pymysql`库，并假设已经建立了与数据库的连接对象`conn`，创建了游标对象`cursor`。

在示例1中，我们使用`cursor.execute()`函数执行了一个简单的SQL查询，获取了所有`customers`表的数据。

在示例2中，我们执行了一个带参数的SQL查询，其中`%s`是参数占位符，我们使用元组`(1,)`作为参数传递给`execute()`函数。

在示例3中，我们执行了多个SQL语句。首先执行了插入操作，将一条新的记录插入到`customers`表中，然后执行了另一个查询操作。

在示例4中，我们使用了字典方式传递参数。SQL语句中的参数名使用`%(param_name)s`的形式，参数值通过字典进行传递。

请注意，在执行完`cursor.execute()`函数后，通常需要使用`fetchone()`、`fetchall()`等函数来获取查询结果。

以上是`cursor.execute()`函数的基本用法和示例。它是执行SQL查询和操作数据库的关键函数之一，可以执行各种类型的SQL语句并传递参数。