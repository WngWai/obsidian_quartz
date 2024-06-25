在 Python 中，`sqlite3.connect()` 函数用于创建一个 SQLite 数据库连接对象。SQLite 是一个轻量级的嵌入式关系型数据库管理系统，`sqlite3` 是 Python 中用于与 SQLite 进行交互的标准库模块。

**功能：** 创建一个 SQLite 数据库连接对象。

**定义：**
```python
sqlite3.connect(database, timeout=5.0, detect_types=0, isolation_level=None, check_same_thread=True, factory=100)
```

**参数介绍：**
- `database`：要连接的 SQLite 数据库文件的路径。如果文件不存在，将创建一个新的数据库文件。
- `timeout`：可选，设置超时时间（秒）。如果在指定的时间内无法获取到数据库连接，则抛出 `sqlite3.OperationalError`。
- `detect_types`：可选，用于启用或禁用类型检测。默认为 0，表示禁用类型检测。
- `isolation_level`：可选，事务隔离级别。可以是 `None`、`"DEFERRED"`、`"IMMEDIATE"` 或 `"EXCLUSIVE"`。默认为 `None`，表示使用 SQLite 默认的隔离级别。
- `check_same_thread`：可选，用于检查连接是否在创建它的线程以外的线程中使用。默认为 `True`，表示启用检查。
- `factory`：可选，用于设置回调函数的参数。默认为 100。

**举例：**
```python
import sqlite3

# 连接到 SQLite 数据库（如果不存在则创建）
conn = sqlite3.connect('example.db')

# 创建一个游标对象
cursor = conn.cursor()

# 执行 SQL 查询
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# 插入数据
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('John Doe', 25))

# 提交更改
conn.commit()

# 关闭连接
conn.close()
```

上述代码演示了如何使用 `sqlite3.connect()` 连接到 SQLite 数据库，创建表，插入数据，并提交更改。在实际使用中，可以根据需要传递不同的参数来配置连接。