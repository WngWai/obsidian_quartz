FROM_UNIXTIME是MySQL中的一个函数，用于将UNIX时间戳转换为日期时间格式。

它接受一个参数，即Unix时间戳（以秒为单位），并返回对应的日期时间值。以下是使用FROM_UNIXTIME函数的示例：

```sql
SELECT FROM_UNIXTIME(unix_timestamp) AS formatted_datetime FROM table_name;
```

在上面的示例中，将unix_timestamp替换为您要转换的Unix时间戳，table_name替换为您的表名。FROM_UNIXTIME函数将Unix时间戳转换为日期时间，并使用别名formatted_datetime将其输出。

如果您希望将转换后的日期时间格式化为特定的日期时间字符串，可以使用DATE_FORMAT函数。以下是一个示例：

```sql
SELECT DATE_FORMAT(FROM_UNIXTIME(unix_timestamp), '%Y-%m-%d %H:%i:%s') AS formatted_datetime FROM table_name;
```

在上面的示例中，`%Y-%m-%d %H:%i:%s`是一个日期时间格式化字符串，用于指定输出的日期时间格式。您可以根据需要修改这个格式化字符串。