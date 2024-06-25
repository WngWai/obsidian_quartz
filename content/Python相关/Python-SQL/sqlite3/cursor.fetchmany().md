在 Python 的数据库编程中，`Cursor.fetchmany()` 函数用于从查询结果中获取指定数量的记录（行）。该方法通常与 `SELECT` 查询一起使用，用于一次性获取多条记录，而不是使用 `fetchall()` 获取所有记录。

**功能：** 从查询结果中获取指定数量的记录（行）。

**定义：**
```python
Cursor.fetchmany(size=None)
```

**参数介绍：**
- `size`：可选参数，要获取的记录数量。如果未指定，将返回默认数量的记录，具体取决于数据库驱动的默认设置。

**返回值：**
返回一个包含获取的记录的列表。

**举例：**
```python
import sqlite3

# 连接到 SQLite 数据库（如果不存在则创建）
conn = sqlite3.connect('example.db')

# 创建一个游标对象
cursor = conn.cursor()

# 创建表
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('John Doe', 25))
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Jane Doe', 30))
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Bob Smith', 22))

# 提交更改
conn.commit()

# 查询数据
cursor.execute("SELECT * FROM users")

# 获取指定数量的记录
result = cursor.fetchmany(size=2)

# 打印查询结果
for row in result:
    print(row)

# 关闭连接
conn.close()
```

在上述代码中，`cursor.fetchmany(size=2)` 用于获取查询结果中的**前两条**记录。这可以帮助在处理大量数据时，逐步获取并处理部分结果，而不必一次性获取所有记录。