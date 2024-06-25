在 Pandas 中，`dtype` 表示数据类型，是 Pandas 中的一个重要概念。每个 Pandas 对象（例如 Series 或 DataFrame）都有一个关联的数据类型，用于指定该对象中的元素的类型。`dtype` 既可以用于描述每列的数据类型（在 DataFrame 中），也可以用于描述一个 Series 对象的数据类型。

看sr.dtype
```python
import pandas as pd

# 创建一个 Series
data = pd.Series([1, 2, 3, 4, 5])

# 获取数据类型
print(data.dtype)

# 输出
int64
```

看df[col].dtype
要查看一个 pandas DataFrame 中某一列的数据格式，可以使用 `dtype` 属性。具体来说，如果 `df` 是一个 DataFrame 变量，`col` 是其一列的字段名，那么可以使用 `df[col].dtype` 来查看该列的数据类型。

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [22, 35, 28],
    'gender': ['female', 'male', 'male'],
    'height': [165.5, 180.0, 175.2]
}
df = pd.DataFrame(data)

print(df['gender'].dtype)

# 输出
object
```

以下是一些常见的 Pandas 数据类型：

1. **数值类型：**
   - `int64`: 64 位整数
   - `float64`: 64 位浮点数

2. **日期和时间类型：**
   - `datetime64`: 日期和时间

3. **布尔类型：**
   - `bool`: 布尔值

4. **字符串类型：**
   - `object`: 通用字符串类型
   - `string`: 字符串类型（Pandas 1.0.0 版本后引入）

5. **分类类型：**
   - `category`: 分类数据类型






