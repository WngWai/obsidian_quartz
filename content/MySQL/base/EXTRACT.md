在MySQL中，`EXTRACT`函数用于从日期或时间值中提取指定的部分，如年、月、日、小时、分钟等。以下是`EXTRACT`函数的语法：

```sql
sql复制代码EXTRACT(field FROM source)
```

其中，`field`是你要提取的部分，可以是以下值之一：

- YEAR
- MONTH
- DAY
- HOUR
- MINUTE
- SECOND

`source`是要从中提取部分的日期或时间值。

以下是一些使用`EXTRACT`函数的示例：

1. 提取年份：
    

```sql
sql复制代码SELECT EXTRACT(YEAR FROM '2023-07-19');
```

输出结果为：2023  
2. 提取月份：

```sql
sql复制代码SELECT EXTRACT(MONTH FROM '2023-07-19');
```

输出结果为：7  
3. 提取日期：

```sql
sql复制代码SELECT EXTRACT(DAY FROM '2023-07-19');
```

输出结果为：19  
4. 提取小时：

```sql
sql复制代码SELECT EXTRACT(HOUR FROM '2023-07-19 14:30:00');
```

输出结果为：14  
5. 提取分钟：

```sql
sql复制代码SELECT EXTRACT(MINUTE FROM '2023-07-19 14:30:00');
```

输出结果为：30  
6. 提取秒数：

```sql
sql复制代码SELECT EXTRACT(SECOND FROM '2023-07-19 14:30:45');
```

输出结果为：45

这些示例演示了如何使用`EXTRACT`函数从日期或时间值中提取不同的部分。你可以根据需要选择适当的字段来提取所需的部分。