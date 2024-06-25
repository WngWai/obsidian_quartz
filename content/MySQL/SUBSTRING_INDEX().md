是 MySQL 中用于**截取字符串**的函数之一。它可以按照指定的分隔符在一个字符串中返回指定数量的子字符串。

```mysql
SUBSTRING_INDEX(str, delim, count)
```
其中：
- `str` 是要截取的**字符串**；
- `delim` 是**分隔符**，在字符串中按此分隔符进行截取；
- `count` 是指定要返回的**子字符串的数量**。如果 `count` 为**正数**，则从字符串的**开头开始**截取；如果 `count` 为**负数**，则从字符串的**末尾开始**截取。

举个简单的例子，假设我们有一个字符串 "Hello, World, How are you?"，我们可以使用 SUBSTRING_INDEX() 函数来截取字符串中的子字符串。例如：

```mysql
SELECT SUBSTRING_INDEX('Hello, World, How are you?', ', ', 2);
```
将返回以下结果：
```mysql
Hello, World
```

这是因为 `SUBSTRING_INDEX()` 函数按照 `', '` 作为分隔符，在给定字符串中找到前两个子字符串，并将它们连接成一个新的字符串作为结果返回。在这个例子中，前两个子字符串是 “Hello” 和 “World”，它们被连接成了 “Hello, World”。所以最终的输出结果就是 “Hello, World”。