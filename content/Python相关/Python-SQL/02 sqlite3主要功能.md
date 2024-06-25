`sqlite3` 模块是 Python 中用于操作 SQLite 数据库的官方模块。SQLite 是一个轻量级的嵌入式数据库引擎，`sqlite3` 模块提供了访问 SQLite 数据库的接口。以下是 `sqlite3` 模块中一些主要的函数和功能，按功能分类介绍：

[[sqlite数据类型]]

pycharm中自带sqlite数据库

```Linux
# 在linux中安装sqlite3
sudo apt install -y sqlite3

# 创建相应数据库文件
sqlite3 db_name.db
```


**conn和cursor**两个关键内容
```python
# -*- codeing = utf-8 -*-
# @Time : 2024/2/12
# @Author : WngWai
# @Software: VScode

"""
利用sqlite3进行sqlite数据库操作。
"""

import pandas as pd
import sqlite3


# 1，连接数据库
conn = sqlite3.connect('sqlite.db') #  查看当前工作目录，print(os.getcwd())。..\sqlite.db找到更上一级的目录了，实际新建了数据库

# 2，创建游标
cursor = conn.cursor()

# 3，执行SQL语句
# 3.1，创建表
sql1 = '''
create table if not exists `book_table`(
    id integer primary key,
    cname text,
    score real
)
''' # create而不是creat
cursor.execute(sql1)

# 3.2，查询
sql2 = '''
select * 
from Book250
order by `id`
'''
cursor.execute(sql2)
rows = cursor.fetchall() # 结果是元组列表

# 打印查询结果
for row in rows:
    print(row)

# 转换为df数据框
df_data = pd.DataFrame(rows, columns = ['id', 'info_link', 'pic_link', 'cname', 'ename', 'score', 'rated', 'introduction', 'info']) # 直接将元组列表转为df数据结构，需要额外指定列名，columns = ['id', ‘info_link‘, ‘pic_link‘, ‘cname‘, ‘ename‘, ‘score‘, ‘rated‘, ‘introduction', 'info']
print(df_data)

# 3.3，插入数据。因为id唯一，所以再次插入同id的内容实际不会插入，并且报错
sql3 = '''
INSERT INTO `book_table`
select id, cname, score 
from `Book250`
where score > 9
'''
cursor.execute(sql3)

# 4，提交事务，关闭游标
conn.commit() # 只是查询就不用了提交了；注意是conn而关闭游标
conn.close()


```
### 1. **连接和关闭数据库：**
conn=[[sqlite3.connect()]] 创建一个数据库连接。**找不到**db文件，就会**新建**一个对应名称的db文件

cursor=[[conn.cursor()|conn.cursor()]]创建一个游标对象，用于执行 SQL 语句。

`conn.close`：关闭数据库连接。

### 2. **执行 SQL 语句：**

[[Python相关/Python-SQL/sqlite3/cursor.execute()|cursor.execute()]] 执行 SQL 语句。

[[Python相关/Python-SQL/sqlite3/cursor.executemany()|cursor.executemany()]]执行多个相同的 SQL 语句，但带有不同参数。

### 3. **获取查询结果：**
**`cursor.fetchone`：** 获取查询结果的**第一行**。

[[Python相关/Python-SQL/sqlite3/cursor.fetchall()|cursor.fetchall()]] 获取查询结果的**所有行**，返回**元组列表**，每个元素都是一个元组，是结构数据中的每行内容

[[Python相关/Python-SQL/sqlite3/cursor.fetchmany()|cursor.fetchmany()]] 获取查询结果的指定数量的行。

### 4. **事务控制：**
[[conn.commit()|conn.commit()]] 对数据库操作（例如新建、插入、更新、删除等），通过调用 commit()方法，你可以将**这些更改永久保存到数据库中**

[[conn.rollback()|conn.rollback()]] 在对数据库操作错误后，返回到上一次commit()后的初始状态

**`Connection.isolation_level`：** 获取或设置事务隔离级别。

### 5. **占位符和参数：**
**占位符 `?`：** 在 SQL 语句中使用**占位符**来表示参数。

 **`:param` 或 `%s` 占位符：** 也可以使用其他类型的占位符。

### 7. **错误处理：**
[[sqlite3.Error]]用这个！异常类，表示与 SQLite 相关的错误。

**`Connection.row_factory` 和 `Cursor.row_factory`：** 设置结果行的工厂函数。



