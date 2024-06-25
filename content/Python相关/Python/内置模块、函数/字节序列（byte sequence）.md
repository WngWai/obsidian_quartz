在Python中，字节序列（byte sequence）是由字节（byte）组成的不可变序列。字节序列表示二进制数据，每个字节可以包含范围在0-255之间的整数值。

字节序列在Python中有两种主要的表示方式：`bytes`和`bytearray`。

1. `bytes`：`bytes`是不可变的字节序列，一旦创建就不能修改。它使用前缀`b`标识，可以使用字节值或转义字符来初始化。例如：`b'Hello'`、`b'\x48\x65\x6c\x6c\x6f'`。 每个字母一个字节，共5个字节。

1. `bytearray`：`bytearray`是可变的字节序列，可以修改其中的字节内容。它与`bytes`相似，但可以通过索引和切片进行修改。例如：`bytearray(b'Hello')`。

字节序列在处理二进制数据、网络通信、文件读写等场景中非常有用。它们可以用来表示图像、音频、视频、加密数据等任意二进制格式的信息。

以下是一个简单的示例，展示如何创建和操作字节序列：

```python
# 创建字节序列
bytes_seq = b'Hello'
print(bytes_seq)  # b'Hello'

# 访问字节序列中的字节
print(bytes_seq[0])  # 72
print(bytes_seq[1])  # 101

# 使用切片获取部分字节序列
slice_seq = bytes_seq[1:4]
print(slice_seq)  # b'ell'

# 尝试修改字节序列（bytes是不可变的，会引发TypeError）
# bytes_seq[0] = 65  # TypeError: 'bytes' object does not support item assignment

# 创建可变字节序列
bytearray_seq = bytearray(b'Hello')
print(bytearray_seq)  # bytearray(b'Hello')

# 修改可变字节序列中的字节
bytearray_seq[0] = 65
print(bytearray_seq)  # bytearray(b'Aello')
```

在这个示例中，我们展示了如何创建字节序列（使用`b`前缀或`bytearray()`函数）。我们还演示了如何访问和操作字节序列中的字节，以及不可变字节序列和可变字节序列之间的区别。