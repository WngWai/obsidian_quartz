`SQLAlchemy` （\[ˈalkəmi\]）是 Python 中流行的 ORM (Object-Relational Mapping) 框架，它提供了一个高级的抽象接口来与数据库进行交互。

SQLAlchemy不再是直接操作SQL语句，而是**操作python对象**。在具体实现代码中，将**数据库表映射成 Python 类和对象**，来简化数据库操作。由于这种转换的存在，使得SQLAlchemy的性能不及原生SQL。

SQLAlchemy提供了**更高层次的抽象和更高级的API**，适用于复**杂的数据库操作和模型映射**。而PyMySQL则提供了**更底层的操作和更简单的API**，适用于简单的数据库操作和直接SQL语句的执行。

跟后面Django中学习的数据库操作框架有些相似！

SQLAlchemy告别手动建模，实现动态写库 - 老乐的文章 - 知乎
https://zhuanlan.zhihu.com/p/597463380
SQLAlchemy 基本操作 - 木土的文章 - 知乎
https://zhuanlan.zhihu.com/p/152607699
SQLAlchemy使用和踩坑记 - 岚岚的文章 - 知乎
https://zhuanlan.zhihu.com/p/466056973

session会话是在ORM层操作的，它提供了面向对象的接口；而conn连接是在数据库API层操作的，它提供了执行SQL语句的接口。

先**创建引擎**，再**创建表**，接着**创建会话对象**进行**表操作**！
```python
"""
利用sqlachemy进行sqlite数据库操作。大致的思路：
1，借助python中现有的数据库驱动，建立数据库引擎，实现数据库的连接；
2，创建Base基类，构建ORM层；
3，创建Session会话类对象，实现高级的ORM层的操作，即用python语言直接操作数据库，允许链式调用。
"""

# 1，进行数据库连接，创建会话对象
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
# 1.1，创建数据库引擎
db_addr = "mysql+pymysql://root:123456@192.168.79.128:3308/mysql_sql"
engine = create_engine(db_addr, poolclass=NullPool) # 定义连接池，避免重复连接和释放，延缓操作

# 2，定义ORM模型
from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, FLOAT, BigInteger
from sqlalchemy.ext.declarative import declarative_base
# 2.1，创建基类，进行定义模型
Base = declarative_base()
# 定义模型，继承基类
class Job(Base):
    # 表名
    __tablename__ = 'job'

	# 表结构
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text)
    status = Column(Integer)

    # 对应的表结构
    #
    # CREATE TABLE IF NOT EXISTS `job` (
    #     `id` INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    #     `name` VARCHAR(255),
    #     `status` INT NOT NULL, 
    # )ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

	# 
    def __repr__(self):
        return "id:%d,name:%s" % (self.id, self.name)

# 创建表，在数据库中创建表
Base.metadata.create_all(engine)

# 3,进行ORM的操作，可以链式调用
from sqlalchemy.orm import sessionmaker # 提供ORM操作函数
from sqlalchemy import func # 推荐这么导入，而非from sqlalchemy.sql import func

# 3.1，创建Session会话对象，进行ORM操作:
Session = sessionmaker(bind=engine)  # 创建Session类对象
session = Session() # 创建Session类的实例对象

# 3.2，INSERT INTO插入操作
item_job1 = Job(id=1, name='1号工作', status='0')
session.add(item_job1)

session.add_all([Job(id=2, name='2号工作', status='0'),
                 Job(id=3, name='2号工作', status='0'),
                 Job(id=4, name='John', status='0')])

session.commit() # 事务不提交的话会存在缓存中，紧跟着查询是查不到内容的

# 3.3，SELECT查询操作  
# 用query()构建查询对象，FROM
item_job2 = session.query(Job).all()  
for job in item_job2:  
    print(job.name, job.status)

# WHERE：筛选，查询所有id大于1的用户  
jobs = session.query(Job).filter(Job.id > 1).all()  
for job in jobs:  
    print(job.name, job.status)

# ORDER_BY：排序，按照年龄升序排序所有用户，然后按名字降序排序。如果年龄相同，那么按照名字升序排序。  
jobs = session.query(Job).order_by(Job.status, Job.name).all()  
for jobs in jobs:  
    print(job.name, job.status)

# GROUP_BY：分组
# jobs = session.query(Job).group_by(Job.name).order_by(Job.id.desc()).all() # 要么分组中带上所有列。要么查询中将未经分组的列进行聚合操作
## 分组中带上所有列
jobs = session.query(Job).group_by(Job.name, Job.id, Job.status).order_by(Job.id.desc()).all()
print(jobs)

## 为经分组的列进行聚合操作
# jobs = session.query(Job.name, func.count(Job.id).label('job_count'), func.max(Job.status).label('latest_status')).group_by(Job.name).order_by(Job.job_count.desc()).all() # 命名别名后，再排序就是各种错误，还得理下
print(jobs)

# 3.4，UPDATA更性操作
# 更新名字为'John'的用户的年龄为35。如果名字不是'John'，那么不进行任何操作。  
session.query(Job).filter_by(name='John').update({"status": 35})  
print(jobs)

# 3.5，DELETE删除操作
# 删除名字为'John'的用户。如果名字不是'John'，那么不进行任何操作。  
session.query(Job).filter_by(name='John').delete()  
print(jobs)

# 4，提交事务，关闭连接
session.commit() # 只有设计数据库改动的操作才提交事务
session.close()
```

### 引擎(Engine)和连接(Connection)
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
```
[[create_engine()]]创建**数据库引擎**实例，它是数据库连接的入口点，通常使用数据库的 URL 来初始化。

### ORM 层(Object-Relational Mapper)
```python
from sqlalchemy.ext.declarative import declarative_base

# 任选一种 
from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, FLOAT, BigInteger
from sqlalchemy.sql.schema mport Column, String, Text, Integer, TIMESTAMP, FLOAT, BigInteger
```
[[Base类对象]]
[[Base = declarative_base()]] 用于**创建基类**，用户定义的 ORM 模型将继承自这个基类。
def Table_name(Base)继承基类，创建模型

`relationship` - 用于定义表之间的关系（如一对多、多对多）。
`backref` - 在关系的另一侧添加反向引用。
`column_property` - 用于定义一个映射到 SQL 表达式的列属性。
`mapper` - 低级的映射器，用于将类关联到表。

[[Base.metadata.create_all(engine)]] 在数据库中建表。Base是基类，代表数据库中的所有表，定义 **(表)模型** 时能继承这个基类；metadata是Base的属性，表示metadata对象，存储关于**数据库表的结构信息**；create_all创建表；engine数据库引擎对象，它提供了**与数据库的连接**。

### 定义表
[[Table]]用于定义和引用数据库中的表。可以通过Table引用现成的表？跟Base模型的区别？
[[Column]] - 表示表中的一列，并指定列的类型，如 Integer、String 等。
[[MetaData]] 容器对象，存储多个 Table 对象的定义信息，每个 `Table` 对象代表数据库中的一个表

`and_`, `or_`, `not_` - 逻辑连接词，用于构建复杂的 WHERE 子句。

### 会话(Session)操作
通过会话对象从而实现用python语言操作数据库。
```python
from sqlalchemy.orm import sessionmaker # 提供ORM操作函数
from sqlalchemy import func # 推荐这么导入，而非from sqlalchemy.sql import func
```

[[Session = sessionmaker(bind=engine)]]

[[sessionmaker()]]用来创建 `Session` 类的实例，后者管理持久化操作的所有对象。
[[Python相关/Python-SQL/sqlalchemy/Session类对象|Session类对象]] 它是实际**执行ORM 操作的对象**，例如添加、修改、删除记录等。

session.commit：事务提交到数据库
session.flush：预提交，提交到数据库文件，还未写入到数据库文件中
session.ollback：session回滚
session.close：session 关闭

session. query()  用于构建查询对象，可以链式调用过滤器、排序等方法。
label()相当于SQL中的as，起**别名**
filter()相当于WHERE**筛选**条件。
filter_by()
order_by() 对结果集进行**排序**
group_by()对结果集进行**分组**，使用[[func]]模块中提供的聚合函数


join() 用于指定 JOIN 操作。

### 迁移(Migration)
`Alembic` - SQLAlchemy 的数据库迁移工具，虽然不是 SQLAlchemy 核心包的一部分，但经常与 SQLAlchemy 一起使用。

### 异常(Exceptions)
SQLAlchemyError**所有SQLAlchemy错误的基类**。
NoResultFound当结果预期**至少有一个元素但没有找到**时抛出。
MultipleResultsFound当结果**预期只有一个元素但找到多个**时抛出。

### 类似pymsql中的操作
[[engine.connect()]] **基于引擎建立一个新的数据库连接**，相当于打开了文件。

[[conn.execute()]]执行SQL语言，需要用text()包住
conn.commit()提交事务
conn.close()关闭sql文件

先**创建引擎**，在创建**连接**，之后跟sqlite3、pymsql库中的操作一样。
```python
"""
利用sqlachemy进行sqlite数据库操作，进行连接操作。
根sqlite3、pymysql有明显的不同：
1，不用创建游标；
2，sql语句需要用text()包裹起来。
"""
import sqlalchemy
import pandas as pd
import pymysql


# 1，进行数据库连接
from sqlalchemy import create_engine, text

# 1.1，创建引擎
db_addr = "mysql+pymysql://root:123456@192.168.79.128:3308/mysql_sql"
engine = create_engine(db_addr)

# 2.2，使用 engine.connect() 创建到数据库的连接，相当于打开数据库
conn = engine.connect()

# 2，执行SQL语句

# 2.1，新建表
conn.execute(text('''
create table if not exists student_table2(
    id int primary key AUTO_INCREMENT,
    name varchar(20),
    age int
)
''')) # 必须带text()

# 2.2，执行查询语句
result = conn.execute(text("SELECT * FROM table_name"))

# 获取查询结果的所有行
rows = result.fetchall()
df_data = pd.DataFrame(rows, columns=['id', 'date', 'name'])
print(df_data)

# 打印结果
for row in rows:
    print(row)

# 提交事务
conn.commit() # 必须是对数据库进行改动才进行在提交事务的操作
# 关闭连接
conn.close()
```