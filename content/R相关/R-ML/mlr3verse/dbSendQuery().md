在 **mlr3db** 包中，`dbSendQuery()` 函数用于向数据库发送 SQL 查询，并返回一个结果集对象。与 `dbGetQuery()` 不同的是，`dbSendQuery()` 不会立即提取结果，而是允许用户逐步提取结果或处理大型查询。

`dbSendQuery()` 函数用于向数据库发送一个 SQL 查询语句，并返回一个结果集对象以供后续处理。适用于需要处理大型数据集或逐步获取查询结果的场景。


- **conn**: 数据库连接对象。通过 `DBI::dbConnect()` 函数建立。
- **statement**: 字符串，表示 SQL 查询语句。
- **...**: 其他传递给方法的参数，这些参数因数据库驱动程序的不同而异。

#### 连接到 SQLite 数据库并发送查询

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

# 发送查询以选择所有记录
query <- "SELECT * FROM my_table"
res <- dbSendQuery(con, query)

# 提取查询结果
data <- dbFetch(res)

# 打印读取的数据
print(data)

# 清理结果对象
dbClearResult(res)

# 断开连接
dbDisconnect(con)
```

### 处理大型查询结果

对于大型查询结果，可以逐步提取数据：

```r
# 发送查询以选择所有记录
query <- "SELECT * FROM my_table"
res <- dbSendQuery(con, query)

# 提取前3条记录
data_part <- dbFetch(res, n = 3)

# 打印部分读取的数据
print(data_part)

# 提取剩余的记录
data_remaining <- dbFetch(res)

# 打印剩余的数据
print(data_remaining)

# 清理结果对象
dbClearResult(res)

# 断开连接
dbDisconnect(con)
```

`dbSendQuery()` 函数提供了一种灵活的方法来向数据库发送 SQL 查询，并逐步处理查询结果。通过这种方式，用户可以更有效地处理大型数据集，避免一次性提取所有数据带来的内存问题。结合 `dbFetch()` 和 `dbClearResult()` 函数，`dbSendQuery()` 可以用于各种复杂的数据库操作和数据分析任务。