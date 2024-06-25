在Python中的`sqldf`包中，`sqldf()`函数用于在DataFrame对象上执行SQL查询。它允许你使用SQL语法对DataFrame数据进行筛选、排序、聚合等操作。

主要函数，用于执行 SQL 查询并返回结果。它接受 SQL 查询语句作为参数，并可选地接受全局变量和局部变量的字典作为参数。

```python
sqldf(query, globals=None, locals=None)
```
参数说明：
- `query`: 必需，表示要**执行的SQL查询语句**，可以是字符串或多行字符串。查询语句应该符合标准的SQL语法。

- `globals`（可选）: 表示**全局**命名空间的字典。默认值为`None`，表示使用当前全局命名空间。

- `locals`（可选）: 表示**局部**命名空间的字典。默认值为`None`，表示使用当前局部命名空间。

```python
from pandasql import sqldf
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}

df = pd.DataFrame(data)
```

现在，我们可以使用`sqldf()`函数执行SQL查询：
```python
# 查询所有行和列
result = sqldf("SELECT * FROM df")
print(result)

# 查询特定列
result = sqldf("SELECT Name, Age FROM df")
print(result)

# 条件筛选
result = sqldf("SELECT * FROM df WHERE Age > 25")
print(result)

# 排序
result = sqldf("SELECT * FROM df ORDER BY Age DESC")
print(result)

# 聚合
result = sqldf("SELECT City, COUNT(*) as Count FROM df GROUP BY City")
print(result)
```

注意：在执行SQL查询时，查询中的表名应与DataFrame对象的变量名保持一致。在查询中，可以使用标准的SQL语法来筛选、排序、聚合等操作。

这些示例展示了如何使用`sqldf()`函数在Python中执行SQL查询，以便在DataFrame数据上进行灵活的操作。

### globals参数
在`pandasql`包中，`sqldf()`函数用于执行SQL查询并返回一个DataFrame。这个函数接受一个名为`globals`的参数，它是一个字典，用于在执行SQL查询时提供全局变量。这些全局变量可以在SQL查询中通过`?`占位符来引用。
以下是`globals`参数的一些关键点：
1. **作用域**：`globals`参数提供了一个作用域，在这个作用域中，SQL查询可以访问和操作变量。
2. **参数化查询**：当你在SQL查询中使用`?`占位符时，`sqldf()`函数会使用`globals`字典中的相应键值对来替换这些占位符。
3. **动态执行**：`globals`字典允许你动态地设置SQL查询中使用的变量，这意味着你可以根据运行时的条件来决定查询的行为。
以下是一个使用`sqldf()`函数和`globals`参数的示例：
```python
import pandas as pd
import pandasql as psql
# 创建一个DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# 定义一个全局变量
global_var = 'global_value'
# 使用sqldf()函数执行一个SQL查询
# 这里使用了?占位符，它将在执行时被globals字典中的值替换
sql_query = "SELECT A, B, ? FROM df"
result = psql.sqldf(sql_query, globals={'?': global_var})
print(result)
```
在这个例子中，我们创建了一个DataFrame，并定义了一个全局变量`global_var`。然后，我们使用`sqldf()`函数执行了一个SQL查询，其中包含了`?`占位符。在执行查询时，我们传递了一个`globals`参数，其中包含了一个键值对，将`?`占位符替换为`global_var`变量的值。
请注意，`sqldf()`函数是`pandasql`库中的一个函数，它允许你执行SQL查询并返回结果作为DataFrame。这个函数接受一个名为`globals`的参数，用于在执行SQL查询时提供全局变量。这些全局变量可以在SQL查询中通过`?`占位符来引用。


### globals和locals的区别？？？
在`pandasql`包中，`sqldf()`函数用于执行SQL查询并返回一个DataFrame。这个函数接受两个参数，`globals`和`locals`，它们都用于在执行SQL查询时提供变量。这两个参数的主要区别在于它们的范围和用途。

 lobals参数
`globals`参数是一个字典，它代表了在执行SQL查询时全局可用的命名空间。在这个作用域中，SQL查询可以访问和操作这些全局变量。这些变量可以通过`?`占位符在SQL查询中引用。`globals`参数适用于传递那些在整个查询中需要多次引用的全局变量。
locals参数
`locals`参数也是一个字典，它代表了在执行SQL查询时局部作用域的命名空间。与`globals`参数相比，`locals`参数中的变量只在查询执行期间有效，并且通常用于传递那些只在查询内部使用的局部变量。

区别
1. **作用域**：
   - `globals`参数定义了全局作用域。
   - `locals`参数定义了局部作用域。
2. **生命周期**：
   - `globals`参数中的变量在整个查询期间都是可用的。
   - `locals`参数中的变量只在查询执行期间有效。
3. **使用场景**：
   - `globals`参数适用于传递那些在整个查询中需要多次引用的全局常量或配置。
   - `locals`参数适用于传递那些只在查询内部使用的临时变量或计算结果。
### 示例
以下是一个使用`sqldf()`函数和`globals`、`locals`参数的示例：
```python
import pandas as pd
import pandasql as psql
# 创建一个DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# 定义全局变量
global_var = 'global_value'
# 定义局部变量
local_var = 'local_value'
# 使用sqldf()函数执行一个SQL查询
# 这里使用了?占位符，它将在执行时被globals字典中的值替换
sql_query = "SELECT A, B, ? FROM df"
# 执行查询，同时提供全局和局部变量
result_globals = psql.sqldf(sql_query, globals={'?': global_var})
result_locals = psql.sqldf(sql_query, locals={'?': local_var})
print(result_globals)
print(result_locals)
```
在这个例子中，我们创建了一个DataFrame，并定义了全局变量`global_var`和局部变量`local_var`。然后，我们使用`sqldf()`函数执行了一个SQL查询，其中包含了`?`占位符。在执行查询时，我们分别提供了`globals`和`locals`参数，它们分别用于传递全局变量和局部变量。
请注意，`sqldf()`函数是`pandasql`库中的一个函数，它允许你执行SQL查询并返回结果作为DataFrame。这个函数接受`globals`和`locals`参数，用于在执行SQL查询时提供变量。这两个参数的主要区别在于它们的范围和用途。
