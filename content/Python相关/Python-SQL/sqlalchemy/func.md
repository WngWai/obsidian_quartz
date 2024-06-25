在 SQLAlchemy 中，`func` 模块提供了一系列用于执行 SQL 聚合操作的函数。这些函数在构建复杂查询时非常有用，尤其是在需要对数据进行分组和汇总的情况下。以下是一些常用的 `func` 模块中的函数及其用途：
1. **`func.count()`**：返回指定列（或表达式）的行数。
   - 示例：`func.count(User.id)` 会返回 `User` 表中 `id` 列的行数。
2. **`func.sum()`**：返回指定列（或表达式）的总和。
   - 示例：`func.sum(Order.amount)` 会返回 `Order` 表中 `amount` 列的总和。
3. **`func.avg()`**：返回指定列（或表达式）的平均值。
   - 示例：`func.avg(User.age)` 会返回 `User` 表中 `age` 列的平均值。
4. **`func.max()`**：返回指定列（或表达式）的最大值。
   - 示例：`func.max(Order.status_date)` 会返回 `Order` 表中 `status_date` 列的最大值。
5. **`func.min()`**：返回指定列（或表达式）的最小值。
   - 示例：`func.min(Order.status_date)` 会返回 `Order` 表中 `status_date` 列的最小值。

**`func.group_concat()`**：返回一个**由逗号分隔的值列表**，这些值来自 GROUP BY 子句中的列。
   - 示例：`func.group_concat(Order.product_name)` 会返回一个由逗号分隔的产品名称列表，每个列表项来自同一个订单。
   
1. **`func.distinct()`**：返回指定列（或表达式）的唯一值列表。
   - 示例：`func.distinct(User.email)` 会返回 `User` 表中 `email` 列的唯一值列表。
2. **`func.now()`**：返回当前的日期和时间，通常用于 `SELECT` 查询中的 `FROM_UNIXTIME()` 函数。
   - 示例：`func.now()` 可以用于获取当前的日期和时间。
3. **`func.lower()`**、`func.upper()`、`func.trim()` 等：这些函数用于对字符串进行转换或修剪。
   - 示例：`func.lower(User.name)` 会返回 `User` 表中 `name` 列的值，转换为小写。
4. **`func.round()`**：对数值进行四舍五入。
    - 示例：`func.round(Order.amount, 2)` 会返回 `Order` 表中 `amount` 列的值，保留两位小数。
这些函数通常与 `group_by` 子句一起使用，以便对结果进行分组和聚合。它们在构建复杂查询时非常有用，尤其是在需要对数据进行汇总和分析时。
在实际使用中，你需要根据具体的查询需求选择合适的函数，并确保它们与你的数据库和 SQL 方言兼容。例如，某些函数可能只在特定类型的数据库中可用。
