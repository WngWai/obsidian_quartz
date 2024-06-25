在 SQLAlchemy 中，`Session` 类对象是 ORM（对象关系映射）会话的实例，它提供了一个与数据库交互的高级接口。以下是一些常用的 `Session` 类对象的属性和方法：

```python
from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 创建数据库引擎和 Session 类
engine = create_engine('sqlite:///:memory:')
Base = declarative_base()

# 定义模型
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50))

# 创建表
Base.metadata.create_all(engine)

# 创建 Session 类
Session = sessionmaker(bind=engine)
session = Session()

# 添加对象到会话
new_user = User(name='John Doe')
session.add(new_user)

# 提交事务
session.commit()

# 查询对象
queried_user = session.query(User).filter_by(name='John Doe').first()
print(queried_user.name)  # Output: 'John Doe'

# 关闭会话
session.close()
```

### 属性：
1. **`query()`**：返回一个 `Query` 对象，用于执行查询。
2. **`bind`**：返回与 `Session` 关联的 `Engine` 对象。
3. **`transaction`**：返回当前的事务对象，如果当前没有事务，则返回 `None`。
4. **`dirty`**：返回一个字典，包含所有被修改但未提交（dirty）的对象。
5. **`new`**：返回一个字典，包含所有新建（new）的对象，这些对象尚未被添加到会话中。
6. **`deleted`**：返回一个字典，包含所有删除（deleted）的对象，这些对象已被标记为删除但尚未从数据库中删除。
### 方法：
1. **`add(instance)`**：将一个实例添加到会话中，该实例将被追踪（tracked）。
2. **`add_all(instances)`**：将多个实例批量添加到会话中。
3. **`delete(instance)`**：删除会话中指定的实例。
4. **`delete_all(instances)`**：删除会话中指定的多个实例。
5. **`flush()`**：将所有更改（添加、修改、删除）的事务刷新到数据库。
6. **`commit()`**：提交当前事务，并执行所有尚未执行的变更。
7. **`rollback()`**：回滚当前事务，撤销所有已提交的变更。
8. **`close()`**：关闭会话，释放与会话关联的所有资源。
9. **`refresh(instance)`**：从数据库中重新加载给定实例的所有数据。
10. **`merge(instance)`**：将一个实例合并到会话中，该实例将成为会话中的一个新实例。
11. **`expire(instance, [all])`**：使给定实例的数据过时，在调用 `query()` 之前，将重新从数据库加载数据。
### 注意事项：
- `Session` 对象必须与一个数据库引擎（`Engine`）绑定，通常是通过 `sessionmaker(bind=engine)` 创建的。
- 操作数据库时，推荐使用事务来管理数据的完整性。你可以使用 `begin()`、`commit()` 和 `rollback()` 方法来管理事务。
- `Session` 对象是线程不安全的，这意味着在一个线程中只能有一个活跃的 `Session` 对象。
使用 `Session` 对象时，你应该遵循 ACID 原则（原子性、一致性、隔离性、持久性），以确保数据的完整性和一致性。
