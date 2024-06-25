是 Pandas 中用于对 DataFrame 进行查询的函数。它提供了一种方便的方式来筛选和过滤 DataFrame 中满足特定条件的数据。

`df.query()` 函数接受一个`字符串`**表达式**作为参数，该字符串表示要查**询的条件**。在查询条件中，你可以使用**列名、逻辑运算符和其他支持的表达式**来过滤数据。

```python
df.query(expr, inplace=False, **kwargs)
```

**参数：**
- `expr`：查询表达式，必需。一个**字符串**，表示要**查询的条件**。
- `inplace`：布尔值，可选。指定是否就地修改原始 DataFrame，默认为 **False**。
- `**kwargs`：关键字参数，可选。用于传递额外的变量给查询表达式。


### 使用外部变量进行查询

假设有一个名为 `df` 的 DataFrame，包含以下数据：
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                   'B': [10, 20, 30, 40, 50],
                   'C': [100, 200, 300, 400, 500]})
```

下面是一些示例，演示如何使用 `df.query()` 进行数据查询：

1. 查询满足特定条件的数据：

   ```python
   result = df.query('A > 2 and B <= 40')
   ```

   输出：

   ```python
      A   B    C
   2  3  30  300
   3  4  40  400
   ```

   在上面的示例中，我们使用查询表达式 `'A > 2 and B <= 40'` 对 DataFrame 进行查询，返回满足条件的数据。

### 使用外部变量进行查询

   ```python
   threshold = 30
   result = df.query('B > @threshold')
   ```

   输出：

   ```python
      A   B    C
   3  4  40  400
   4  5  50  500
   ```

   在此示例中，我们通过在查询表达式中使用 `@` 前缀引用外部变量 `threshold`，对 DataFrame 进行查询。查询结果为 `B` 列中大于 `threshold` 变量的所有数据。

### 使用字符串方法进行查询：

   ```python
   result = df.query('A.astype(str).str.contains("2")', engine='python')
   ```

   输出：

   ```python
      A   B    C
   1  2  20  200
   ```

   在这个示例中，我们使用查询表达式 `'A.astype(str).str.contains("2")'` 实现了字符串匹配。查询结果为 `A` 列中包含数字 2 的所有数据。

除了上面的示例，`df.query()` 函数还支持使用逻辑运算符、比较运算符、算术运算符、函数调用等。你可以根据需要编写符合条件的查询表达式。


### 对datatime也是有效

```python
import pandas as pd
import numpy as np
import datetime

list_of_dates = [
    "2019-11-20",
    "2020-01-02",
    "2020-02-05",
    "2020-03-10",
    "2020-04-16",
    "2020-05-01",
]
employees = ["Hisila", "Shristi", "Zeppy", "Alina", "Jerry", "Kevin"]
salary = [200, 400, 300, 500, 600, 300]
df = pd.DataFrame(
    {"Name": employees, "Joined_date": pd.to_datetime(list_of_dates), "Salary": salary}
)

filtered_df = df.query("Joined_date >= '2019-06-1' and Joined_date <='2020-02-05'")
print(filtered_df)

# 输出
      Name Joined_date  Salary
0   Hisila  2019-11-20     200
1  Shristi  2020-01-02     400
2    Zeppy  2020-02-05     300
```
