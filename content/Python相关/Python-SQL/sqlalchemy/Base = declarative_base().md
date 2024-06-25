`declarative_base` 是 SQLAlchemy 中的一个函数，用于**创建一个基类，通过这个基类可以定义数据库模型类**，使其更加简洁、易读。

```python
sqlalchemy.ext.declarative.declarative_base(cls=<class 'sqlalchemy.orm.declarative.api.Base'>, name='Base', metadata=None, class_registry=None)
```

- **`cls`：**
  - 描述：要继承的基类。
  - 类型：`type`。
  - 默认值：`sqlalchemy.orm.declarative.api.Base`。

- **`name`：**
  - 描述：创建的基类的名称。
  - 类型：字符串。
  - 默认值：`'Base'`。

- **`metadata`：**
  - 描述：关联的 `MetaData` 对象。
  - 类型：`MetaData` 对象。
  - 默认值：`None`。

- **`class_registry`：**
  - 描述：类注册表，用于跟踪所有继承自 `Base` 的类。
  - 类型：`TypeRegistry` 对象。
  - 默认值：`None`。

### 示例：

```python
from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base

# 创建数据库引擎
engine = create_engine('sqlite:///:memory:')

# 创建声明性基类
Base = declarative_base()

# 定义模型
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50))

# 创建表
Base.metadata.create_all(engine)

# 使用模型创建对象
new_user = User(name='John Doe')

# 访问模型属性
print(new_user.name)  # Output: 'John Doe'
```

在这个示例中，首先创建了一个 SQLite 内存数据库引擎。然后，使用 `declarative_base` 函数创建了一个声明性基类 `Base`。接着，定义了一个简单的 `User` 模型，通过继承 `Base` 类，定义了表名和列。使用 `Base.metadata.create_all(engine)` 创建了表。最后，通过模型创建了一个用户对象，并通过访问模型的属性展示了模型的使用。