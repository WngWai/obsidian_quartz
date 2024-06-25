pandas中用于数据表**转置**的属性，T是transpose的缩写，表示转置。这个属性可以直接被应用在pandas的**DataFrame**和**Series**对象上，将它们的行与列对调，可以使得数据表的对应关系更加清晰明确。

下面是一个使用`.T`进行数据表转置的例子：

```python
import pandas as pd

# 创建一个数据表
data = {'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'gender': ['F', 'M', 'M']}
df = pd.DataFrame(data)

# 输出数据表
print(df)

# 转置数据表
df_transpose = df.T

# 输出转置后的数据表
print(df_transpose)
```

输出结果如下：

```python
       name  age gender
0    Alice   25      F
1      Bob   30      M
2  Charlie   35      M

             0    1        2
name    Alice  Bob  Charlie
age        25   30       35
gender      F    M        M
```

可以看到，原先的数据表`df`包含了3行3列的数据，其中包含姓名、年龄和性别3个字段，而转置之后生成的数据表`df_transpose`则变成了3列3行的数据表，而每一行则对应着原先数据表中的每一列，具体来说，第一行是姓名这个字段对应的3个取值（即每个人的姓名），第二行是年龄这个字段对应的3个取值，第三行是性别这个字段对应的3个取值。这样转置后，我们得到了以字段作为行的新数据表，更容易进行一些聚合和分析操作。