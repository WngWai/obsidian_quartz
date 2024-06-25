在 Python 中，`len()` 是一个内置函数，用于**返回对象的长度或元素个数**。它适用于字符串、列表、元组、字典、集合等可迭代对象。
```python
len(obj)
``` 
  - `obj`：必需，要计算长度的对象。

1. 计算字符串的长度：
   ```python
   text = "Hello, World!"
   length = len(text)
   print(length)
   # 输出: 13
   ```

2. 计算列表的长度：
   ```python
   my_list = [1, 2, 3, 4, 5]
   length = len(my_list)
   print(length)
   # 输出: 5
   ```

3. 计算元组的长度：
   ```python
   my_tuple = (1, 2, 3, 4, 5)
   length = len(my_tuple)
   print(length)
   # 输出: 5
   ```

4. 计算字典中键的数量：
   ```python
   my_dict = {'a': 1, 'b': 2, 'c': 3}
   length = len(my_dict)
   print(length)
   # 输出: 3
   ```

5. 计算集合中元素的数量：
   ```python
   my_set = {1, 2, 3, 4, 5}
   length = len(my_set)
   print(length)
   # 输出: 5
   ```

`len()` 函数返回的是对象中元素的个数或长度。对于字符串，它返回字符串的字符数；对于列表、元组、字典和集合，它返回其中元素的数量。

需要注意的是，对于字典，`len()` 函数计算的是字典中键的数量，而不是键值对的数量。如果需要计算键值对的数量，可以使用 `len(my_dict.items())`。
