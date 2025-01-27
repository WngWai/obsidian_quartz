在Python中，字符串（str）类型有一个名为 `find()` 的方法，用于确定该字符串是否包含子字符串，如果包含，则返回子字符串的第一次出现的索引；如果不包含，则返回 -1。
### 定义：
```python
str.find(sub[, start[, end]])
```
### 参数介绍：
- **sub**：要查找的子字符串。
- **start** (可选)：搜索的起始位置，默认为0。
- **end** (可选)：搜索的结束位置，默认为字符串的长度。
### 返回值：
- 如果找到子字符串，则返回子字符串的第一次出现的索引。
- 如果未找到子字符串，则返回 -1。
### 应用举例：
```python
text = "Hello, world!"
# 查找子字符串
index = text.find("world")
# 打印查找结果
print(index)  # 输出: 7
```
在这个例子中，`find()` 方法在字符串 `text` 中查找子字符串 "world"，并返回子字符串的起始索引，即7。
`find()` 方法通常用于确定字符串中是否包含某个子字符串，以及该子字符串的起始位置。如果需要找到所有匹配项，可以使用 `count()` 方法，或者使用 `search()` 方法，该方法返回第一个匹配项的索引，如果没有找到匹配项，则返回 -1。
