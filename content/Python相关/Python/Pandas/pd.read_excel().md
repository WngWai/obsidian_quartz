 是 Pandas 中用于读取 Excel 文件的函数。下面是一些常用的参数介绍：

- `io`：必填参数，文件的路径或 URL。
- `sheet_name`：可选参数，读取的 **sheet 的名称或索引**。默认为 0，即读取第一个 sheet。
- `header`：可选参数，指定**哪一行为列名**。默认值为 0，即第一行为列名。
- `index_col`：可选参数，设置**哪一列为行索引**。默认为 `None`，无行索引。

- `usecols`：可选参数，要**读取的列**。可以是**列名称或列序号**。例如，`usecols=['A', 'C', 'E']` 或 `usecols=[0, 2, 4]`。
接收一个整数（表示列的索引，从 0 开始计数），或者是一个列的索引或名字列表，或者是一个可调用对象。

- `skiprows`：可选参数，**跳过的行数**。默认为 0，跳过前面的行。

- `dtype`：可选参数，指定**每一列的数据类型**。可以是 Python 的数据类型或 NumPy 的数据类型。必须是字典！


- `names`：可选参数，指定**列名**。如果指定了 `header=None`，则需要指定列名。
- `na_values`：可选参数，指定 **NaN 值的标记**，将它们视为**缺失值**。在参数中NULL也是缺失值！

缺失值在SQL中不是'NULL'值，所以需要fillna将缺失值替换下！
```python
# 0，读取excel中的交易卡号和交易对手账卡号  
df_card = pandas.read_excel(path, usecols=['交易卡号', '交易对手账卡号'], dtype=str, na_values='NULL')  
# SQL语句不识别nan值，所以指定nan值转化为字符串'null'  
df_card = df_card.fillna('NULL')
```

首先需要安装 `openpyxl` 模块，因为 Pandas 不支持 Excel 2013 之前的版本：

```python
!pip install openpyxl
```

然后我们可以读取一个简单的 Excel 文件：

```python
import pandas as pd

df = pd.read_excel('example.xlsx', sheet_name='Sheet1', header=0)
print(df)
```

在这个例子中，我们读取了 `example.xlsx` 文件中的 `Sheet1`，并将第一行作为列名。

###  index_col参数举例
可以用于指定读取 Excel 文件时用作 DataFrame 的索引列的列号或列名。

如果需要使用某个 Excel 列作为 DataFrame 的行索引，可以将该列的列名或列号传递给 `index_col` 参数。下面是一个示例：

假设我们有一个包含以下内容的 Excel 文件 `example.xlsx` ：

| time                | temperature | humidity |
|---------------------|-------------|----------|
| 2021-06-15 00:00:00 | 26.8        | 71       |
| 2021-06-15 01:00:00 | 26.5        | 72       |
| 2021-06-15 02:00:00 | 26.3        | 74       |

我们可以使用以下代码将该 Excel 文件导入为 DataFrame，并将 "time" 列设置为行索引：

```python
import pandas as pd

df = pd.read_excel('example.xlsx', index_col='time')
print(df)
```

输出：

```python
                     temperature  humidity
time                                      
2021-06-15 00:00:00         26.8        71
2021-06-15 01:00:00         26.5        72
2021-06-15 02:00:00         26.3        74
```

在上面的代码中，我们将 "time" 列作为 DataFrame 的行索引。此时，行索引为 "2021-06-15 00:00:00"、"2021-06-15 01:00:00" 和 "2021-06-15 02:00:00"，并且列索引为 "temperature" 和 "humidity"。


### 用usecols=[]可以在录入时实现列的排序录入
是的，使用`usecols=[]`可以按照指定的列顺序读取Excel文件并生成DataFrame。

例如，如果我们有一个Excel文件，其中包含三列数据：姓名(name)、年龄(age)、性别(gender)，我们可以通过以下代码读取它并按照指定的顺序生成DataFrame：

```python
import pandas as pd

# 指定要读取的列顺序
cols = ['name', 'age', 'gender']

# 读取Excel文件，并仅保留指定的列
df = pd.read_excel('data.xlsx', usecols=cols)

print(df)
```

输出结果为：

```python
    name  age gender
0    Tom   25      F
1   Jack   30      M
2  Alice   40      F
```

在上面的例子中，我们通过`cols`变量指定了要读取的列顺序，并将其传递给了`usecols`参数。由于仅保留了指定的列，因此最终生成的DataFrame中的列顺序就按照`cols`中指定的顺序进行排列。

### 参数扩展
#### 参数dtype举例
可以只**指定其中几列**的数据类型，而不必为所有列都指定数据类型。甚至可以举几个不在表中的列名，在适用函数调用时挺合适的！

在`pd.read_excel()`函数的`dtype`参数中，你可以仅指定你感兴趣的列的数据类型，而对其他列不进行指定，Pandas将会根据数据内容自动推断其数据类型。

以下是一个示例，演示如何使用`dtype`参数来指定Excel文件中各列的数据类型：

```python
import pandas as pd

# 指定每列的数据类型，格式必须是字典

dtypes = {
    '交易日期': str,
    '交易卡号': str,
    '交易金额': float,
    '交易类型': str,
}

# 读取Excel文件并指定数据类型
df = pd.read_excel('data.xlsx', dtype=dtypes)

# 输出DataFrame的数据类型
print(df.dtypes)
```

在上述示例中，我们假设要读取的Excel文件名为"data.xlsx"，并且包含以下列：'交易日期'、'交易卡号'、'交易金额'和'交易类型'。我们使用`dtype`参数来指定每列的数据类型。
在这个例子中，我们将'交易日期'和'交易卡号'列指定为字符串类型（str），将'交易金额'列指定为浮点数类型（float），将'交易类型'列指定为字符串类型（str）。


### 实际中遇到的问题
#### 读取的空值为NAN，SQL不识别
```python
df_card = pandas.read_excel(path, usecols=['交易卡号', '交易对手账卡号'], dtype=str)  
# SQL语句不识别nan值，所以指定nan值转化为字符串'null'  
df_card = df_card.fillna('NULL')
```