在 Python 的数据库编程中，`Cursor.execute()` 函数用于执行 SQL 语句。这函数通常由数据库游标对象调用，游标对象是通过连接对象的 `cursor()` 方法创建的。

**功能：** 执行 SQL 语句。

**定义：**
```python
Cursor.execute(operation, parameters=None)
```

**参数介绍：**
- `operation`：要执行的 SQL 语句字符串。
- `parameters`：可选参数，用于传递 SQL 语句中的参数。参数可以是单个值，也可以是一个元组或字典，具体取决于 SQL 语句中是否使用参数占位符。

**返回值：**
该方法没有返回值，但执行 SQL 语句对数据库产生影响。

**举例：**
```python
import sqlite3

# 连接到 SQLite 数据库（如果不存在则创建）
conn = sqlite3.connect('example.db')

# 创建一个游标对象
cursor = conn.cursor()

# 创建表.
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('John Doe', 25))

# 查询数据
cursor.execute("SELECT * FROM users")
result = cursor.fetchall()

# 打印查询结果
for row in result:
    print(row)

# 提交更改
conn.commit()

# 关闭连接
conn.close()
```

上述代码演示了如何使用 `Cursor.execute()` 方法执行创建表、插入数据、查询数据等 SQL 操作。`fetchall()` 方法用于获取查询结果。在实际应用中，根据具体需要，可以执行不同的 SQL 语句，如更新数据、删除数据等。



### 如果SQLite中已经相应表
1，SQLite not support update column.

**如果已有表**，再次执行创建会报错！
如果你想**更改表中的列的数据类型或属**性，你需要重新创建这个表。这不是一个直接更新列的操作，而是一个**删除旧表并创建一个新表**的过程。
