用于控制查询结果集的返回行数的关键字。它可以接受两个参数，分别是起始位置和要返回的行数。

语法如下：
```mysql
SELECT 列名 FROM 表名
LIMIT [起始位置,] 返回行数;
```

其中，起始位置是可选的，表示从哪一行开始返回结果，默认是从第一行开始。返回行数指定要返回的行数。
其实位置**0**表示第一行
假设有一张名为`employees`的表，包含以下数据：

| emp_id | emp_name | emp_salary |
|--------|----------|------------|
| 1      | John     | 5000       |
| 2      | Lisa     | 6000       |
| 3      | Mike     | 7000       |
| 4      | Sarah    | 5500       |
| 5      | Tom      | 4500       |
1. 返回前两行数据：
```sql
SELECT * FROM employees
LIMIT 2;
```

| emp_id | emp_name | emp_salary |
|--------|----------|------------|
| 1      | John     | 5000       |
| 2      | Lisa     | 6000       |

2. 返回从第三行开始的两行数据：
```sql
SELECT * FROM employees
LIMIT 2, 2;
```

| emp_id | emp_name | emp_salary |
|--------|----------|------------|
| 3      | Mike     | 7000       |
| 4      | Sarah    | 5500       |

3. 返回从第四行开始的所有数据：
```sql
SELECT * FROM employees
LIMIT 3, 9999;
```

| emp_id | emp_name | emp_salary |
|--------|----------|------------|
| 4      | Sarah    | 5500       |
| 5      | Tom      | 4500       |
