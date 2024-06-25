`CAST` 是 SQL 中的一个函数，用于将一个数据类型转换为另一个数据类型。它允许你明确地将一个数据类型的值转换为另一个数据类型。

基本语法是：

```sql
sql复制代码CAST (expression AS type[length])
```

其中：

- `expression` 是要转换的值或列。
- `type` 是要转换到的数据类型。
- `[length]` 是可选的，表示数据类型的长度或大小。不是所有的数据类型都需要这个参数。

例如：

1. 将整数列转换为字符列：
    

```sql
sql复制代码SELECT CAST(int_column AS CHAR) FROM table_name;
```

2. 将字符列转换为日期：
    

```sql
sql复制代码SELECT CAST(char_column AS DATE) FROM table_name;
```

3. 将字符列转换为整数：
    

```sql
sql复制代码SELECT CAST(char_column AS INT) FROM table_name;
```

注意：在进行数据类型转换时，必须确保源数据是可以成功转换到目标数据类型的。例如，尝试将包含字母的字符串转换为整数可能会导致错误。