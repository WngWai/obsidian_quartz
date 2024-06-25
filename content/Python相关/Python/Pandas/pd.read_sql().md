**功能：**
`pd.read_sql()` 函数是 pandas 库中的一个函数，用于从 SQL 查询、表格或数据库中读取数据，并返回一个 DataFrame 对象。

**所属包：**
pandas

**定义：**
```python
pandas.read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None, chunksize=None)
```

**参数介绍：**
- `sql`：字符串，**SQL 查询语句**或**表名**。

- `con`：**数据库连接对象**，可以是 SQLAlchemy 的 `Engine`数据库引擎对象或 SQLite3 的数据库文件名。

- `index_col`：指定 DataFrame 中的列用作索引。默认为 `None`。

- `coerce_float`：如果为 True（默认值），将尝试**将非字符串、非数字的列强制转换为浮点型**。

- `params`：字典或参数元组，用于**传递给 SQL 查询的参数**。

- `parse_dates`：指定**要解析为日期的列**。默认为 `None`。

- `columns`：指定**要选择的列**。默认为 `None`，表示选择所有列。

- `chunksize`：指定**每次获取的行数**。如果设置了该参数，返回一个迭代器，每次迭代返回指定行数的 DataFrame。

**举例：**
```python
import pandas as pd
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

# 定义 SQL 查询语句
sql_query = 'SELECT * FROM your_table'

# 使用 pd.read_sql 从数据库中读取数据
df = pd.read_sql(sql_query, engine)

# 打印 DataFrame
print(df)
```

**输出：**
这个示例中的输出将是从数据库中读取的数据的 DataFrame。

在上述示例中：
- `pd.read_sql(sql_query, engine)` 从数据库中读取数据，并返回一个 DataFrame。
- `sql_query` 包含要执行的 SQL 查询语句。
- `engine` 是 SQLAlchemy 的 `Engine` 对象，表示数据库连接。

`pd.read_sql()` 是一个强大的函数，它允许你轻松地从数据库中读取数据并将其转换为 pandas DataFrame，方便进行数据分析和处理。