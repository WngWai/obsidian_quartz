```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   姓名     3 non-null      object
 1   年龄     3 non-null      int64 
 2   性别     3 non-null      object
dtypes: int64(1), object(2)
memory usage: 200.0+ bytes
```

- 数据集**类型**为`pandas.core.frame.DataFrame`
- 数据集共有**3行**，索引是**0 to 2**

- 共有3列，详细的内容：

	数据集有**3个列**，分别为“姓名”、“年龄”、“性别”
	每列中有**非空数据**的数量
	每列的数据类型，其中“姓名”和“性别”为对象（字符串类型），而“年龄”为整数类型

- 整个数据集占用了大约200字节的内存空间。

是pandas库中DataFrame对象的一个**方法**。该方法返回一个文字形式的关于DataFrame的简要描述，包括列名、非空数据的数量、数据类型和内存占用等信息，以帮助初步了解**数据集的结构**。


```python
import pandas as pd

# 创建一个DataFrame对象
data = {'姓名': ['小明', '小红', '小张'], '年龄': [18, 20, 22], '性别': ['男', '女', '男']}
df = pd.DataFrame(data)

# 使用info()方法显示数据集的信息
df.info()
```

上述代码创建了一个包含姓名、年龄和性别信息的DataFrame对象，并使用`info()`方法显示了数据集的基本信息。运行以上代码输出如下：


