在Python中，字符串（str）类型有一个名为 `join()` 的方法，用于将序列中的元素以指定的分隔符连接成一个字符串。


- 如果要连接的元素不是字符串类型，需要先进行转换。例如：`' '.join(map(str, numbers))`，其中 `numbers` 是整数列表。


### 定义：
```python
str.join(iterable)
```
### 参数介绍：
- **iterable**：一个序列，如列表（list）、元组（tuple）或其他可迭代对象。
### 返回值：
- 返回一个字符串，其中包含序列中的元素，元素之间由调用字符串（即第一个参数）分隔。
### 应用举例：
```python
# 创建一个列表
lst = ['apple', 'banana', 'cherry']
# 使用 join() 方法将列表中的元素连接成一个字符串
result = ' '.join(lst)
print(result)  # 输出: apple banana cherry
# 使用不同的分隔符
result = '-'.join(lst)
print(result)  # 输出: apple-banana-cherry
```
在这个例子中，我们首先创建了一个包含三个字符串元素的列表 `lst`。然后，我们使用 `join()` 方法将列表中的元素连接成一个字符串。第一个参数 `' '` 表示空格，用于分隔列表中的元素。如果使用不同的分隔符，如 `'-'`，则会得到不同的连接结果。
`join()` 方法通常用于将多个字符串或其他可迭代对象中的元素合并成一个单一的字符串。


### 进阶
在 Python 中，`join()` 方法通常是字符串对象的方法，而不是函数。根据**指定内容将字符串序列中的元素连接在一起，返回一个新的字符串**。以下是关于字符串的 `join()` 方法的基本信息：

```python
sql = ''' 
insert into Book250 (info_link, pic_link, cname,ename, score, rated, introduction, info) values(%s)''' % ",".join(str_list)
```

%前面是**带有占位符**的字符串内容，后面是**待插入**的内容。
`% ",".join(data)` 将 `",".join(data)` 的结果插入到 `values(%s)` 的位置。这意味着，如果 `data` 是 `['value1', 'value2', 'value3']`，那么 `",".join(data)` 将返回 `'value1,value2,value3'`，并且这个结果将被插入到 `values()` 的位置。

