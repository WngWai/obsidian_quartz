在 Python 的数据库编程中，`Connection.rollback()` 函数用于回滚（撤销）自上一次 `commit()` 以来对数据库所做的更改。如果在执行一系列数据库操作时发生错误或者出现问题，你可以调用 `rollback()` 方法，将数据库的状态还原到上一次成功提交的状态。

**功能：** 回滚（撤销）自上一次 `commit()` 以来对数据库所做的更改。

**定义：**
```python
Connection.rollback()
```

**参数介绍：**
该方法没有参数。

**返回值：**
该方法没有返回值。

**举例：**
```python
import sqlite3

try:
    # 连接到 SQLite 数据库（如果不存在则创建）
    conn = sqlite3.connect('example.db')

    # 创建一个游标对象
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

    # 插入数据
    cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('John Doe', 25))

    # 模拟一个错误，例如数据验证失败
    raise ValueError("Simulated error")

    # 提交更改
    conn.commit()

except Exception as e:
    # 发生异常时，回滚更改
    print(f"Error: {e}")
    conn.rollback()

finally:
    # 关闭连接
    conn.close()
```

在上述代码中，通过模拟一个错误（例如，数据验证失败），引发了一个 `ValueError` 异常。在异常处理块中，`conn.rollback()` 被调用，将数据库的状态回滚到上一次成功提交的状态。这确保了在发生错误时不会对数据库产生永久影响。在实际应用中，你可以根据需要添加适当的异常处理逻辑。