SQLite支持多种数据类型，包括数字、文本、日期和时间、BLOB（二进制大对象）等。以下是一些在SQLite中常用的数据类型名称：
1. 数字类型：
   - `INTEGER`：整数类型，可以是正数、负数或零。 **int**
   - `REAL`：浮点数类型，用于存储带有小数的数值。**float**
   - `FLOAT`：与`REAL`类型等价，通常用于与旧代码或标准SQL兼容。
   - `DOUBLE`：与`REAL`类型等价，通常用于与旧代码或标准SQL兼容。
   - `NUMERIC`：与`REAL`类型等价，通常用于与旧代码或标准SQL兼容。
2. 文本类型：
   - `TEXT`：用于存储字符串数据，没有长度限制。 **varchar**
   - `CHAR`：固定长度的字符串，但SQLite中不推荐使用`CHAR`类型，通常使用`TEXT`类型。
3. 日期和时间类型：
   - `DATE`：日期值，格式为`YYYY-MM-DD`。
   - `TIME`：时间值，格式为`HH:MM:SS`，也可以包含日期部分`YYYY-MM-DD HH:MM:SS`。
   - `DATETIME`：日期和时间值，格式为`YYYY-MM-DD HH:MM:SS`。
   - `TIMESTAMP`：时间戳值，格式为`YYYY-MM-DD HH:MM:SS.SSS`。
4. 二进制数据类型：
   - `BLOB`：用于存储任意二进制数据。
5. 特殊类型：
   - `NULL`：表示值是未知的或未定义的。
   - `BOOLEAN`：表示真（`TRUE`）或假（`FALSE`）。
请注意，SQLite中的`INTEGER PRIMARY KEY`是一个特殊的整数类型，它还具有主键的特性。如果表中有一个`INTEGER PRIMARY KEY`列，SQLite会自动为主键列生成一个唯一且自动增加的值。
在创建表时，您可以根据数据的特性和需求选择合适的数据类型。例如，如果您的分数数据是整数，那么应该使用`INTEGER`类型；如果是带有小数的数值，那么应该使用`REAL`类型。如果数据是文本，比如书名或人名，那么应该使用`TEXT`类型。
