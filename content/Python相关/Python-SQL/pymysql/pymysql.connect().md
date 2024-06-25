在Python中，`pymysql.connect()`函数用于与MySQL数据库建立连接并返回一个连接对象。
```python
pymysql.connect(
    host=None,           # 数据库服务器地址或主机名
    user=None,           # 数据库登录用户名
    password='',         # 用户登录密码
    database=None,       # 数据库名
    port=0,              # 数据库服务器的端口号，默认为 3306
    unix_socket=None,    # UNIX 套接字文件路径，用于 UNIX 系统上的本地连接
    charset='',          # 连接使用的字符集，默认为 ''
    sql_mode=None,       # SQL 模式设置
    read_default_file=None, # 指定 my.cnf 配置文件的路径
    conv=None,           # 类型转换字典
    use_unicode=None,    # 是否使用 Unicode 字符集，默认为 True
    client_flag=0,       # MySQL 客户端标志
    cursorclass=pymysql.cursors.Cursor, # 游标类型，默认为 Cursor 类
    init_command=None,   # 在建立连接后，但在其他设置之前执行的 SQL 语句
    connect_timeout=10,  # 连接超时时间，默认为 10 秒
    ssl=None,            # SSL 参数字典
    read_default_group=None, # my.cnf 文件中的组名
    compress=None,       # 是否使用压缩协议，默认为 None
    named_pipe=None,     # 是否使用命名管道
    autocommit=False,    # 是否自动提交，默认为 False
    db=None,             # 数据库名，与 database 参数相同
    passwd=None,         # 密码，与 password 参数相同
    local_infile=False,  # 是否允许使用 LOAD DATA LOCAL INFILE，默认为 False
    max_allowed_packet=16*1024*1024, # 最大允许的数据包大小，默认为 16MB
    defer_connect=False, # 是否延迟建立连接，默认为 False
    auth_plugin_map={},  # 身份验证插件映射
    read_timeout=None,   # 读取超时时间
    write_timeout=None,  # 写入超时时间
    bind_address=None,   # 绑定地址
    binary_prefix=False, # 是否使用二进制前缀
)

```


**函数定义**：
```python
pymysql.connect(host='localhost', user='root', password=None, database='孝感', port=3306, charset=None)
```

**参数**：
以下是`pymysql.connect()`函数中的参数：

- `host`：MySQL服务器主机名或IP地址。默认为`None`，表示连接本地主机。

- `user`：MySQL数据库的用户名。

- `password`：MySQL数据库的密码。

- `database`：要连接的数据库名称。默认为`None`，表示不连接到特定数据库。

- `port`：MySQL服务器的端口号。默认为`None`，表示使用默认端口号。得是**整数**

- `charset`：字符集。默认为`None`，表示使用MySQL服务器的默认字符集。

**示例**：
以下是使用`pymysql.connect()`函数建立与MySQL数据库的连接的示例：

```python
import pymysql

# 示例1：连接本地MySQL服务器
conn1 = pymysql.connect(host='localhost', user='root', password='password', database='mydatabase')

# 示例2：连接远程MySQL服务器
conn2 = pymysql.connect(host='example.com', user='myuser', password='mypassword', database='mydatabase', port=3306, charset='utf8')

# 示例3：连接本地MySQL服务器，并指定字符集
conn3 = pymysql.connect(host='localhost', user='root', password='password', database='mydatabase', charset='utf8mb4')
```

在上述示例中，我们首先导入了`pymysql`库。

在示例1中，我们使用`pymysql.connect()`函数连接到本地MySQL服务器。我们指定了主机名为`localhost`，用户名为`root`，密码为`password`，要连接的数据库为`mydatabase`。函数返回一个连接对象`conn1`，可以用于执行SQL查询和操作数据库。

在示例2中，我们使用`pymysql.connect()`函数连接到远程MySQL服务器。我们指定了主机名为`example.com`，用户名为`myuser`，密码为`mypassword`，要连接的数据库为`mydatabase`，端口号为`3306`，字符集为`utf8`。函数返回一个连接对象`conn2`。

在示例3中，我们使用`pymysql.connect()`函数连接到本地MySQL服务器，并指定了字符集为`utf8mb4`。字符集的选择取决于具体的需求和数据库配置。函数返回一个连接对象`conn3`。

请注意，连接MySQL数据库需要正确的主机名、用户名、密码和可用的数据库。具体的连接参数和配置取决于您的实际情况。

```python
import pymysql.cursors

# 连接到数据库
connection = pymysql.connect(host='example.com',  # 数据库服务器的地址
                             user='user',         # 数据库登录用户名
                             password='password', # 用户密码
                             database='mydb',     # 数据库名
                             charset='utf8mb4',   # 使用的字符集
                             cursorclass=pymysql.cursors.DictCursor)  # 返回字典形式的结果集

# 使用连接
try:
    with connection.cursor() as cursor:
        # 创建一条新的记录
        sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        cursor.execute(sql, ('webmaster@example.com', 'very-secret'))

    # 提交本次插入的记录
    connection.commit()

    with connection.cursor() as cursor:
        # 查询插入的记录
        sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
        cursor.execute(sql, ('webmaster@example.com',))
        result = cursor.fetchone()
        print(result)
finally:
    # 关闭数据库连接
    connection.close()

```


