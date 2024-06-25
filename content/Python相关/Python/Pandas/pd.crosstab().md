是pandas库中的一个函数，用于创建**交叉表**（或称为列联表）。交叉表是一种用于展示**两个或多个变量之间关系**的统计表格。


**语法：**
```python
pd.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)
```

**参数说明：**
- `index`：指定交叉表的**行索引**，可以是一个Series或一个列表，表示需要进行分组的变量。
- `columns`：指定交叉表的**列索引**，可以是一个Series或一个列表，表示需要统计的变量。
- `values`：可选参数，指定要计算的值的列，可以是一个Series或一个列表，表示需要进行计算的变量。
- `rownames`：可选参数，用于指定行索引的名称。
- `colnames`：可选参数，用于指定列索引的名称。
- `aggfunc`：可选参数，用于指定对值进行**聚合的函数**，默认为`None`，表示不进行聚合。常用的聚合函数有`sum`、`mean`、`count`等。
- `margins`：可选参数，指定是否在结果中添加行和列的汇总，默认为`False`，表示不添加。
- `margins_name`：可选参数，用于指定行和列汇总的名称。
- `dropna`：可选参数，指定是否移除包含缺失值的行，默认为`True`，表示移除。
- `normalize`：可选参数，指定是否对结果进行归一化，默认为`False`，表示不进行归一化。

**返回值：**
返回一个DataFrame对象，代表交叉表。

**示例：**
假设我们有一个关于学生考试成绩的数据集，包含学生的性别（gender）、科目（subject）和分数（score）信息，我们可以使用`pd.crosstab()`函数创建交叉表，用于分析不同性别学生在各科目上的分数情况。

```python
import pandas as pd

# 创建示例数据
data = {
    'gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female'],
    'subject': ['Math', 'English', 'Math', 'English', 'Math', 'English'],
    'score': [80, 90, 85, 95, 75, 88]
}
df = pd.DataFrame(data)

# 创建交叉表
cross_tab = pd.crosstab(df['gender'], df['subject'], values=df['score'], aggfunc='mean')

print(cross_tab)
```

输出结果：
```python
subject  English  Math
gender                
Female      91.5  85.0
Male        90.0  77.5
```

以上示例中，我们传递了两个Series作为`index`和`columns`，并指定了`values`为分数列，`aggfunc`为`mean`表示计算平均值。结果是一个交叉表，展示了不同性别学生在不同科目上的平均分数。

### 交叉表和透视表的区别
交叉表主要用于对分类型变量之间的交叉计数进行统计分析，而透视表则是在交叉计数的**基础上提供更多的汇总统计指标**，更便于数据探索和进行多维分析
