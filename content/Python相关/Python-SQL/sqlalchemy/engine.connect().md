**功能：**
`engine.connect()` 函数用于**创建到数据库的连接**。这个连接对象**可以执行各种 SQL 操作**，包括查询、插入、更新等。

**所属包：**
SQLAlchemy 库

**定义：**
```python
engine.connect()
```

**参数介绍：**
该函数没有额外的参数。

**举例：**
```python
from sqlalchemy import create_engine

# 定义数据库连接信息
db_username = 'your_username'
db_password = 'your_password'
db_host = 'your_host'
db_port = 'your_port'
db_name = 'your_database'

# 创建数据库连接字符串
db_connection_str = f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'

# 使用 create_engine 创建数据库引擎
engine = create_engine(db_connection_str)

# 使用 engine.connect() 创建到数据库的连接，相当于打开数据库
connection = engine.connect()

# 执行 SQL 查询
result = connection.execute("SELECT * FROM your_table")

# 获取查询结果的所有行
rows = result.fetchall()

# 打印结果
for row in rows:
    print(row)

# 关闭连接
connection.close()
```

**输出：**
这个示例中的输出将取决于执行的具体 SQL 查询，通常是查询结果的行。

在上述示例中：
- `engine.connect()` 创建到数据库的连接，并返回一个 `Connection` 对象。
- `connection.execute()` 方法用于执行 SQL 查询语句。
- `result.fetchall()` 方法获取查询结果的所有行。
- 最后，通过 `connection.close()` 关闭连接。

建议在实际应用中使用 `with` 语句管理连接的上下文，以确保在操作完成后及时释放连接资源。