阶段性只是看下数据！

```python
from pandasql import sqldf
```

`pandasql` 是一个 Python 库，它允许用户使用 SQL 语法来查询 pandas DataFrame。这个库非常有用，尤其是对于那些熟悉 SQL 但不熟悉 pandas 的高级数据操作功能的用户。

已有的数据集：
[[sqldf.load_meat()]]从 PandasSQL 自带的**示例数据库**中加载肉类消费数据集
sqldf.load_births()从 PandasSQL 自带的示例数据库中加载 `births` 表的数据。
sqldf.load_ipl()从 PandasSQL 自带的示例数据库中加载 `ipl` 表的数据。


[[sqldf()]]在**df数据结构上执行 SQL 查询语句** 
### 基本查询
- `SELECT`：选择 DataFrame 中的列。
```python
df_data1 = sqldf('SELECT name, age FROM df_data')
```
- `WHERE`：过滤 DataFrame 中的行。
```python
df_data1 = sqldf('SELECT * FROM df_data WHERE age > 30')
```
- `GROUP BY`：对 DataFrame 中的行进行分组。
```python
df_data1 = sqldf('SELECT city, COUNT(*) FROM df_data GROUP BY city')
```
- `ORDER BY`：对查询结果进行排序。
```python
df_data1 = sqldf('SELECT * FROM df_data ORDER BY age DESC')
```
### 聚合函数
- `COUNT()`：计算行数。
- `SUM()`：计算数值列的总和。
- `AVG()`：计算数值列的平均值。
- `MAX()`：找到数值列的最大值。
- `MIN()`：找到数值列的最小值。
```python
df_data1 = sqldf('SELECT city, COUNT(*) as count, SUM(age) as total_age FROM df_data GROUP BY city')
```
### 连接（JOIN）
- `JOIN`：将两个 DataFrame 根据指定的键合并。
```python
df_data2 = pd.DataFrame({'city': ['New York', 'Los Angeles'], 'population': [8600000, 3900000]})
df_data1 = sqldf('SELECT df_data.name, df_data.age, df_data2.population FROM df_data JOIN df_data2 ON df_data.city = df_data2.city')
```
### 子查询
- 子查询可以在 SQL 查询内部使用，通常用于更复杂的筛选或聚合。
```python
df_data1 = sqldf('SELECT * FROM df_data WHERE age > (SELECT AVG(age) FROM df_data)')
```
### 别名（AS）
- `AS`：为列或表指定别名。
```python
df_data1 = sqldf('SELECT name AS student_name, age AS student_age FROM df_data')
```
### 窗口函数（WINDOW）
- 窗口函数可以对 DataFrame 中的数据进行复杂的数据分析。
```python
df_data1 = sqldf('SELECT name, age, RANK() OVER (ORDER BY age DESC) as age_rank FROM df_data')
```
请注意，`pandasql` 支持大多数标准的 SQL 功能，但它可能不支持所有 SQL 方言的特定功能。此外，`pandasql` 的性能可能不如直接使用 pandas 的内置函数和语法，因为后者是为 Python 和 pandas 优化的。因此，对于复杂的 DataFrame 操作，建议优先使用 pandas 的内置功能。




