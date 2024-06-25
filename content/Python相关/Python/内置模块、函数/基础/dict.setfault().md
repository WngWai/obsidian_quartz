如果键存在,则返回该键对应的值。如果键不存在,则将键-值对添加到字典中,并返回默认值。
```python
dict.setdefault(key, default=None)
```
- `key`: 要获取的键。
- `default`: 如果键不存在,则设置此值作为键的默认值。默认为 `None`。

如果 key 存在于字典中,则**返回该 key 对应的值**。
如果 key 不存在,则创建该 key 并将其值设置为 default (默认为 None),然后**返回 default**。
类似字典的索引，dict\['key'\]


1. **初始化字典**:
```python
my_dict = {}
my_dict.setdefault('name', 'John')
print(my_dict)  # 输出: {'name': 'John'}
```
如果 `'name'` 键不存在,则设置默认值为 `'John'`。

2. **添加多个值到同一个键**:
```python
num_index = {}
num_index.setdefault(1, []).append(10) # num_index[1]
num_index.setdefault(1, []).append(20) # num_index[1]
num_index.setdefault(2, []).append(30)
print(num_index)  # 输出: {1: [10, 20], 2: [30]}
```
利用 `setdefault()` 可以方便地构建 "键-多值" 的映射关系。

3. **检查键是否存在**:
```python
my_dict = {'apple': 5, 'banana': 3}
value = my_dict.setdefault('apple', 10)
print(value)  # 输出: 5

value = my_dict.setdefault('orange', 8)
print(value)  # 输出: 8
print(my_dict)  # 输出: {'apple': 5, 'banana': 3, 'orange': 8}
```
如果键存在,`setdefault()` 返回该键对应的值;如果键不存在,则设置默认值并返回默认值。

