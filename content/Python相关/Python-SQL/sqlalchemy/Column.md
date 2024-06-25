在SQLAlchemy中，`Column()` 函数用于定义表结构中的列。它是一个构建器，用于创建表示数据库表列的对象。这些列对象最终会被用来创建实际的表结构。

### 定义和参数介绍：
`Column()` 函数的基本定义如下：
```python
def Column(name, type, *args, **kwargs):
    ...
```
- **name**：必需参数，表示**列的名称**。
- **type**：必需参数，表示列的**数据类型**。它可以是SQLAlchemy的数据类型，如`Integer`、`String`、`Float`等，也可以是字符串，表示数据库特定的数据类型。
- **args**：可选参数，传递给列类型构造函数的其他参数。
- **kwargs**：可选参数，包含列的额外属性，如：
  - `nullable`：布尔值，表示列**是否可以为NULL**。
  - `default`：**列的默认值**。
  - `primary_key`：布尔值，如果为True，表示该列是**表的主键**。
  - `unique`：布尔值，如果为True，表示该列的**值必须是唯一**的。
  - `index`：布尔值，如果为True，表示**在列上创建索引**。
  - `autoincrement`：布尔值或整数，如果为True或一个整数，表示该列是**自动增长的**。
  
### 应用举例：
以下是一些使用 `Column()` 函数的基本示例：

**定义一个简单的列：**
```python
from sqlalchemy import Column, Integer, String
# 定义一个列，名为 'id'，数据类型为 Integer，为主键
id = Column('id', Integer, primary_key=True)
```
**定义一个带默认值的列：**
```python
from sqlalchemy import Column, Integer, String
# 定义一个列，名为 'age'，数据类型为 Integer，默认值为 18
age = Column('age', Integer, default=18)
```
**定义一个带索引的列：**
```python
from sqlalchemy import Column, Integer, String
# 定义一个列，名为 'email'，数据类型为 String，并创建一个索引
email = Column('email', String, index=True)
```
**定义一个带自动增长标识的列：**
```python
from sqlalchemy import Column, Integer, String
# 定义一个列，名为 'serial_number'，数据类型为 Integer，自动增长
serial_number = Column('serial_number', Integer, autoincrement=True)
```

在定义模型类时，通常会使用 `Column()` 函数来为每个表列创建一个属性。这些属性最终会被用来创建数据库表的定义。例如：
```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    
    id = Column('id', Integer, primary_key=True)
    name = Column('name', String)
    email = Column('email', String)
```
在这个例子中，`User` 类定义了三个列：`id`、`name` 和 `email`，它们分别映射到数据库表中的同名列。
