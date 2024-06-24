在 **mlr3db** 包中，`dbReadTable()` 函数用于从数据库中读取整个表格到 R 数据框。这是一个便捷的函数，适用于需要将数据库中的完整表格数据导入 R 环境的场景。

`dbReadTable()` 是一个读取数据库表格的函数，通过指定数据库连接和表名，可以将数据库中的表格数据读取到 R 数据框中。


- **conn**: 数据库连接对象。这是通过 `DBI::dbConnect()` 函数建立的连接。
- **name**: 字符串，表示要读取的数据库表名。
- **...**: 其他传递给 `DBI::dbReadTable()` 的参数，通常不需要特别指定。


以下是使用 `dbReadTable()` 函数的示例，展示如何从 SQLite 和 PostgreSQL 数据库中读取表格数据。

#### 连接到 SQLite 数据库并读取表格

```r
# 加载必要的包
library(DBI)
library(RSQLite)
library(mlr3db)

# 建立与 SQLite 数据库的连接
con <- dbConnect(RSQLite::SQLite(), dbname = ":memory:")

# 创建一个示例数据框并写入数据库
df <- data.frame(id = 1:5, value = letters[1:5])
dbWriteTable(con, "my_table", df)

# 从数据库中读取表格数据
data <- dbReadTable(con, "my_table")

# 打印读取的数据
print(data)

# 断开连接
dbDisconnect(con)
```

#### 连接到 PostgreSQL 数据库并读取表格

```r
# 加载必要的包
library(DBI)
library(RPostgres)
library(mlr3db)

# 建立与 PostgreSQL 数据库的连接
con <- dbConnect(RPostgres::Postgres(),
                 dbname = "my_database",
                 host = "localhost",
                 port = 5432,
                 user = "my_username",
                 password = "my_password")

# 创建一个示例数据框并写入数据库
df <- data.frame(id = 1:5, value = letters[1:5])
dbWriteTable(con, "my_table", df)

# 从数据库中读取表格数据
data <- dbReadTable(con, "my_table")

# 打印读取的数据
print(data)

# 断开连接
dbDisconnect(con)
```
