在 Pandas 中，`DataFrame.between()` 函数用于返回满足指定条件的数据子集。该函数允许你按照条件选择 DataFrame 中某列的值在指定范围内的行。

以下是 `DataFrame.between()` 函数的基本信息：

**功能：** 返回满足指定条件的数据子集。

**定义：**
```python
DataFrame.between(left, right, inclusive=True, axis=None)
```

**参数介绍：**
- `left`：范围的左侧边界。
- `right`：范围的右侧边界。
- `inclusive`：可选参数，指定是否包括左右边界。默认为 `True`，表示包括左右边界；如果设置为 `False`，则不包括左右边界。
- `axis`：可选参数，指定应用条件的轴。默认为 `None`，表示应用于整个 DataFrame。

**返回值：**
返回一个 DataFrame，其中包含满足条件的行。

**举例：**
```python
import pandas as pd

# 创建一个示例 DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# 选择列 'A' 中在范围 [2, 4] 内的行
result = df[df['A'].between(2, 4)]

# 打印结果
print(result)
```

**输出：**
```
   A   B
1  2  20
2  3  30
3  4  40
```

在这个例子中，`df['A'].between(2, 4)` 返回一个布尔索引，然后用于选择 DataFrame 中满足条件的行。最终的结果是包含列 'A' 中在范围 [2, 4] 内的行的新 DataFrame。

### 时间上的使用
```python
import pandas as pd
import numpy as np
import datetime

list_of_dates = [
    "2019-11-20",
    "2020-01-02",
    "2020-02-05",
    "2020-03-10",
    "2020-04-16",
    "2020-05-01",
]
employees = ["Hisila", "Shristi", "Zeppy", "Alina", "Jerry", "Kevin"]
salary = [200, 400, 300, 500, 600, 300]
df = pd.DataFrame(
    {"Name": employees, "Joined_date": pd.to_datetime(list_of_dates), "Salary": salary}
)

filtered_df = df.loc[df["Joined_date"].between("2019-06-1", "2020-02-05")]
print(filtered_df)

# 输出
      Name Joined_date  Salary
0   Hisila  2019-11-20     200
1  Shristi  2020-01-02     400
2    Zeppy  2020-02-05     300
```