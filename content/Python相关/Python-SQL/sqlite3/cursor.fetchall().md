如果你是在处理 SQLite 数据库，那么 `cursor.fetchall()` 方法通常是在执行 SELECT 查询后，从数据库游标中提取所有行的方法。

**返回一个列表**，列表中的每个元素都是一个**元组**，代表一行数据。

### `cursor.fetchall()` 方法介绍：

```python
cursor.fetchall()
```

下面是一个使用 `sqlite3` 模块的简单示例，演示如何执行 SELECT 查询并使用 `fetchall()` 获取所有结果：

```python
import sqlite3

try:
    # 连接到数据库
    connection = sqlite3.connect(':memory:')  # 使用内存数据库作为示例

    # 创建游标对象
    cursor = connection.cursor()

    # 创建表
    cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')

    # 插入数据
    cursor.execute('INSERT INTO users (name) VALUES (?)', ('John',))
    cursor.execute('INSERT INTO users (name) VALUES (?)', ('Alice',))

    # 执行查询
    cursor.execute('SELECT * FROM users')

    # 获取所有结果
    rows = cursor.fetchall()

    # 打印结果
    for row in rows:
        print(row)

except sqlite3.Error as e:
    print("SQLite Error:", e)

finally:
    # 关闭数据库连接
    if connection:
        connection.close()
```


### 将查询结果转换为DataFrame
```python
df = pd.DataFrame(rows, columns=['id', 'name', 'salary'])
```

