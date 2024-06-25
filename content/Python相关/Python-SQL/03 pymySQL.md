`PMySQL` 库的函数可以根据它们的功能来划分，主要集中在数据库连接、执行命令、事务处理以及结果获取等方面。下面是根据这些功能划分的常用函数：

pymysql中可以执行创建数据库操作，但不建议，因为要在开始连接到指定数据库中，方便后续执行sql语句。

```python
import pymysql

# 数据库连接
connection = pymysql.connect(host='localhost', user='user', password='password', database='dbname')

try:
    # 创建游标
    with connection.cursor() as cursor:
        # 执行SQL查询
        sql = "SELECT `id`, `name` FROM `users`"
        cursor.execute(sql)
        
        # 获取所有查询结果
        results = cursor.fetchall()
        for row in results:
            print(row)
    
    # 提交事务
    connection.commit()
except Exception as e:
    # 出现错误时回滚
    connection.rollback()
    print(e)
finally:
    # 关闭连接
    connection.close()
```

### 数据库连接
[[pymysql.connect()]]连接到MySQL数据库服务器

### 创建游标
[[connection.cursor()]]创建一个**游标对象**，通过该对象来执行查询和获取结果。

[[Cursor类对象]] 在sqlite3中也是类似的功能
### 执行SQL语句
[[Python相关/Python-SQL/pymysql/cursor.execute()]]执行单个SQL。
[[Python相关/Python-SQL/pymysql/cursor.executemany()]]执行多个SQL语句，通常用于批量插入。

### 结果获取
[[cursor.fetchone()]]从结果中获取下一行数据。
[[Python相关/Python-SQL/pymysql/cursor.fetchmany()]]返回结果中的多行数据，`size` 参数指定要返回的记录数。
[[Python相关/Python-SQL/pymysql/cursor.fetchall()]]从结果中获取所有行，返回的结果是**元组列表**

### 游标和连接的关闭
[[cursor.close()]] 关闭游标对象。当你完成了对游标的所有操作后，应该关闭它以释放与它相关的资源，下次执行语句时又重新创建游标，从头开始搜索！

conn.commit()提交事务
[[connection.close()]]关闭数据库连接。

### 事务处理
[[connection.commit()]]提交当前事务，使自上次提交/回滚以来对数据库所做的更改成为永久的。
[[connection.rollback()]]回滚当前事务，撤销自上次提交/回滚以来对数据库所做的更改。

### 辅助功能
[[cursor.description]]获取查询结果的列信息。
[[connection.autocommit()]]设置数据库连接的自动提交模式。

