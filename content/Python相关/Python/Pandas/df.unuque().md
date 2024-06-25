是一个pandas DataFrame对象的方法，用于返回该DataFrame中每一列的唯一值。这个方法返回一个包含每列唯一值的Numpy**一维数组**，**其中每个唯一值只出现一次，且按照出现顺序排列**。如果DataFrame中有缺失值NaN，它们也会被视为唯一值并返回。这个方法通常用于数据清洗和数据探索，可以帮助我们快速了解数据中有哪些不同的取值。

我来给你一个例子。假设我们有一个简单的DataFrame，包含学生姓名和他们的成绩：

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'Grade': [85, 90, 80, 85, 95]}

df = pd.DataFrame(data)
```

这个DataFrame长这样：

```python
       Name  Grade
0     Alice     85
1       Bob     90
2   Charlie     80
3     David     85
4     Emily     95
```

现在我们可以使用`df.unique()`方法，来查看每一列的唯一值：

```python
>>> df.unique()
array(['Alice', 'Bob', 'Charlie', 'David', 'Emily', 85, 90, 80, 95])
```

这个方法返回了一个包含每列唯一值的Numpy数组，其中姓名列的唯一值是'Alice', 'Bob', 'Charlie', 'David', 'Emily'，成绩列的唯一值是85, 90, 80, 95。需要注意的是，这个方法会**将所有列的唯一值合并在一起，并按照它们在DataFrame中出现的顺序排列**。如果我们只想查看某一列的唯一值，可以使用`df['column_name'].unique()`方法。例如，如果我们只想查看成绩列的唯一值，可以这样做：

```python
>>> df['Grade'].unique()
array([85, 90, 80, 95])
```