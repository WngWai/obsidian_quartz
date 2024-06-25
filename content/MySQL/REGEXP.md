是SQL中常见的字符串匹配函数之一，它用于在文本中查找符合特定正则表达式模式的字符串。下面是对REGEXP函数的详细介绍和一些示例：
   ```mysql
   REGEXP(expr, pattern)
   ```

 - `expr`：要进行模式匹配的**字符串表达式**。
 
 - `pattern`：要匹配的**正则表达式模式**。

正则表达式模式语法：
   正则表达式是一种灵活且强大的模式匹配方法，可以根据需求编写不同的模式。这里只提供一些常见的模式元字符用法示例：
   - `.`：匹配除换行符以外的任意字符。
   - `*`：匹配前面的字符零次或多次。
   - `+`：匹配前面的字符一次或多次。
   - `?`：匹配前面的字符零次或一次。
   - `[abc]`：匹配字符 a、b 或 c 中的任意一个。
   - `[a-z]`：匹配小写字母 a 到 z 中的任意一个。
   - `[0-9]`：匹配数字 0 到 9 中的任意一个。
   - `\`：转义字符，用于匹配特殊字符。

1. `.`（点）：匹配任意单个字符，除了换行符。
 示例：`a.b` 可以匹配 "aab"、"acb" 等，但不匹配 "ab" 或 "a\nb"。
2. `*`：匹配前面的元素零次或多次。
示例：`ab*c` 可以匹配 "ac"、"abc"、"abbc" 等。
3. `+`：匹配前面的元素一次或多次。
示例：`ab+c` 可以匹配 "abc"、"abbc"、"abbbc" 等，但不匹配 "ac"。
4. `?`：匹配前面的元素零次或一次。
 示例：`ab?c` 可以匹配 "ac" 或 "abc"，但不匹配 "abbc"。
5. `[]`：定义一个字符集，匹配其中的任意一个字符。
 示例：`[abc]` 可以匹配 "a"、"b" 或 "c"。   
6. `[^]`：在字符集中使用 "^"，表示匹配除了字符集中指定的字符之外的任意字符。
示例：`[^abc]` 可以匹配除了 "a"、"b" 和 "c" 之外的任意字符。
7. `()`：标记一个子表达式，可以改变操作符的优先级，也可以用于提取匹配的子字符串。 
示例：`(abc)+` 可以匹配 "abc"、"abcabc"、"abcabcabc" 等。
8. `|`：表示逻辑上的"或"，匹配两个选择中的任意一个。
 示例：`apple|orange` 可以匹配 "apple" 或 "orange"。
9. `^`：匹配输入的开始位置。
 示例：`^abc` 可以匹配以 "abc" 开头的字符串。 
10. `$`：匹配输入的结束位置。
示例：`abc$` 可以匹配以 "abc" 结尾的字符串。


假设我们有一个名为 `products` 的表，其中包含 `name` 列，现在我们想要查询所有以字母 "A" 开头的产品名称。

SQL查询语句可以这样写：
   ```sql
   SELECT * FROM products WHERE name REGEXP '^A'
   ```

   这个查询将返回所有以字母 "A" 开头的产品名称。

   另一个示例，假设我们有一个名为 `customers` 的表，其中包含 `email` 列，现在我们想要查询所有以 ".com" 结尾的电子邮件地址。

   SQL查询语句可以这样写：
   ```sql
   SELECT * FROM customers WHERE email REGEXP '\.com$'
   ```

   这个查询将返回所有以 ".com" 结尾的电子邮件地址。

总的来说，REGEXP函数提供了一种便捷的方式来进行复杂的字符串模式匹配操作，可以在SQL语句中灵活地应用于各种字符串查询需求。注意，具体的语法和支持的正则表达式元字符可能会因不同的SQL数据库而有所差异，你需要查阅相应数据库的文档以获取更准确的细节。


### 多个关键字
如果您有多个关键字需要同时进行模糊查询，可以使用正则表达式的"或"条件 `(pattern1|pattern2|pattern3)` 来匹配多个关键字。以下是示例代码：

```MySQL
SELECT * 
FROM `724调单的银行交易流水`
WHERE `交易方户名` LIKE '%金木%' AND (`对手户名` LIKE '%建设%' OR `对手户名` LIKE '%房%')

SELECT * 
FROM `724调单的银行交易流水`
WHERE `交易方户名` LIKE '%金木%' AND `对手户名` REGEXP '建设|房'
```


```sql
SELECT column1, column2, ...
FROM table_name
WHERE column1 REGEXP '(keyword1|keyword2|keyword3)';
```

在上面的示例中，`REGEXP`运算符使用括号和竖线将多个关键字分组，表示匹配其中任何一个关键字。您可以根据需要增加或删除关键字，并在括号中添加更多模式。

请注意，在某些数据库中，您可能需要使用双反斜杠 `\\` 来转义正则表达式中的特殊字符。如果关键字包含正则表达式的元字符（如`.`、`+`、`*`等），则需要进行转义。

```sql
SELECT column1, column2, ...
FROM table_name
WHERE column1 REGEXP '(key word1|key\\.word2|key\\+word3)';
```

这将匹配包含 `'key word1'`、`'key.word2'` 或者 `'key+word3'` 的值。记得根据您所使用的数据库和特定的转义规则进行适当的转义和调整。