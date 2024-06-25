在SQLAlchemy中，`declarative_base()` 函数用于创建一个基类，该基类可以用来定义模型类，这些模型类将映射到数据库表。这个基类通常被称为 `Base` 类，它继承自 `sqlalchemy.ext.declarative.DeclarativeMeta`。
### 常用属性：
- `metadata`：基类的 `metadata` 属性是一个 `MetaData` 对象，它包含数据库表的结构信息。你可以通过这个属性来配置表的选项，如表名、索引等。
### 常用方法：
- `__abstractmethods__`：这个属性是一个集合，包含所有在基类中声明但尚未在子类中实现的方法名称。
- `__subclasses__()`：返回一个包含所有子类的列表。
- `__init_subclass__(cls, **kwargs)`：这是一个类方法，当子类被创建时会自动调用。在这个方法中，你可以进行一些自定义的初始化操作，如设置默认的 `metadata`。
- `__table_args__`：一个类属性，用于指定创建表时应使用的额外参数。
- `__table_class__`：一个类属性，用于指定用于创建表的类。默认情况下，它被设置为 `Table`。
- `__tablename__`：一个类属性，用于指定表的名称。如果未指定，表名将与类名相同。
- `query`：一个类方法，用于创建一个 `Query` 对象，该对象可以用来查询与该基类相关联的表。
- `metadata`：一个类属性，用于指定 `MetaData` 对象，该对象包含表的元数据。
### 示例：
```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
```
在这个示例中，`Base` 类是使用 `declarative_base()` 函数创建的，它包含了一个名为 `User` 的子类。`User` 类映射到数据库中的 `users` 表，并且定义了三个列：`id`、`name` 和 `email`。
在创建模型类时，通常会使用 `Column` 类来定义表的列，`Integer` 和 `String` 是列的数据类型。`primary_key=True` 参数表示 `id` 列是表的主键。
