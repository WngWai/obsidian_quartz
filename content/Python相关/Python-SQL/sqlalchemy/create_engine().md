`create_engine()` 函数是 SQLAlchemy 库中的一个重要函数，用于**创建数据库引擎**。数据库引擎是 SQLAlchemy 中**与数据库通信的核心对象**，它负责**管理数据库连接池、执行 SQL 语句**等操作。以下是 `create_engine()` 函数的详细信息：

**函数签名：**
```python
sqlalchemy.create_engine(url, **kwargs)
```

**参数介绍：**
- `url`：数据库连接字符串，包含有关数据库连接的信息，如**数据库类型、用户名、密码、主机地址、端口号**等。
- `**kwargs`：可选的关键字参数，用于指定其他配置选项，例如连接池的大小、字符集等。

数据库连接池：将对象存储起来，避免重复连接、释放，占用过多内存！

数据库连接池原理？ - lemon Tree的回答 - 知乎
https://www.zhihu.com/question/39920723/answer/2353776414

**返回值：**
- 返回一个 `Engine` 对象，表示与数据库的连接。

**示例：**
```python
from sqlalchemy import create_engine

# 定义 MySQL 数据库连接信息
db_username = 'your_username' # 用户名
db_password = 'your_password' # 密码
db_host = 'your_host' # 主机地址
db_port = 'your_port' # 端口号
db_name = 'your_database' # SQL数据库名称

# 创建 MySQL 数据库连接字符串
db_connection_str = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'

# 使用 create_engine 创建数据库引擎
engine = create_engine(db_connection_str)

# 打印 Engine 对象
print(engine)
```

```python
engine = create_engine("mysql+pymysql://root:XXXXXX@localhost:3306/sql")

engine = create_engine('sqlite:///your_database.db') 
```

SQLAlchemy需要使用其他的python的**数据库驱动**来连接数据库，python的数据库驱动有：
![[Pasted image 20240306100920.png]]
```python
MySQL-Python 只支持 myqsl 3.23- 5.5 的版本和python 2.4-2.7版本
    mysql+mysqldb://<user>:<password>@<host>[:<port>]/<dbname>
​
mysql-connector-python 同时支持python2和python3。纯python实现，性能没有MySQL-python好，但支持连接池，线程安全，可靠性高
    mysql+mysqlconnector://<user>:<password>@<host>[:<port>]/<dbname>

支持python3，纯python实现，但不支持线程安全
pymysql 
    mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]
​

cx_Oracle
    oracle+cx_oracle://user:pass@host:port/dbname[?key=value&key=value...]
```


### 参数的具体内容
`create_engine()` 函数中的 `**kwargs` 参数提供了很多配置选项，以下是其中一些常见的参数，不包含具体的例子：

1. **pool_size (int):**
   指定连接池的大小，即同时可以有多少个连接。默认是 5。
2. **pool_recycle (int):**
   指定连接在连接池中的超时时间，以秒为单位。默认是 3600 秒（1小时）。
3. **echo (bool):**
   如果设置为 `True`，将打印引擎发出的**所有 SQL 语句**，用于调试。
4. **execution_options (dict):**
   用于传递执行选项的字典，影响所有执行的语句。
5. **isolation_level (str):**
   指定数据库连接的隔离级别，例如 `'READ COMMITTED'`。
6. **connect_args (dict):**
   用于传递给底层数据库连接的额外参数的字典。
7. **max_overflow (int):**
   允许连接池中的连接数超过 `pool_size` 的最大值。默认为 `10`。
8. **encoding (str):**
   指定连接的字符编码。
9. **convert_unicode (bool):**
   如果设置为 `True`，将使用 Unicode 进行字符串编码。
10. **echo_pool (bool):**
如果设置为 `True`，将在连接池中发生事件时输出调试信息。
11. **pool_timeout (float):**
指定从连接池获取连接的超时时间，以秒为单位。默认为 `30` 秒。

### mysql+mysqlconnector和mysql+pymysql的差异
`mysql+mysqlconnector` 和 `mysql+pymysql` 是 SQLAlchemy 中用于连接 MySQL 数据库的两个不同的数据库连接字符串格式。它们使用了不同的 MySQL 驱动。

1. **`mysql+mysqlconnector`：**
   - 使用 `mysql-connector-python` 作为 MySQL 驱动。
   - `mysql-connector-python` 是 MySQL 官方提供的驱动。
   - 使用时需要安装 `mysql-connector-python` 模块。
   - 连接字符串示例：`mysql+mysqlconnector://user:password@localhost/db`

2. **`mysql+pymysql`：**
   - 使用 `PyMySQL` 作为 MySQL 驱动。
   - `PyMySQL` 是一个纯 Python 实现的 MySQL 客户端库。
   - 使用时需要安装 `pymysql` 模块。
   - 连接字符串示例：`mysql+pymysql://user:password@localhost/db`

选择使用哪个连接字符串主要取决于你的需求和个人偏好。通常来说，两者都可以用于连接 MySQL 数据库，但可能有些微妙的性能和功能差异。