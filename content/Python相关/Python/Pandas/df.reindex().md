```python
import pandas as pd

# 创建一个简单的DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['a', 'b', 'c'])

# 重排索引和列
new_index = ['c', 'b', 'a', 'd'] # 'd' 是新引入的标签
new_columns = ['C', 'A', 'B', 'D'] # 'D' 是新引入的列
df_reindexed = df.reindex(index=new_index, columns=new_columns, fill_value=0)

print(df_reindexed)

### 输出
   C  A  B  D
c  9  3  6  0
b  8  2  5  0
a  7  1  4  0
d  0  0  0  0
```

在Python中，`df.reindex()`方法是用来**调整DataFrame的行标签（index）和列标签（columns）**。直接重新排序了，新标签名也会添加新的数据

**函数定义**：
```python
df.reindex(labels=None, index=None, columns=None, axis=None, method=None, copy=True)
```

下面是 `DataFrame.reindex()` 方法的一些常用参数的详细说明：

1. **index**:
   - 新的行标签（index）。
   - 你可以传递一个索引列表，DataFrame会按照这个新索引进行**重排**。如果在新索引中存在原DataFrame中没有的标签，则在结果DataFrame中这些位置会被设置为`NaN`。

2. **columns**:
   - 新的列标签。
   - 类似于行标签，传递一个新列的列表，DataFrame会**重排**列，不存在的列将会被添加并填充为`NaN`。
目前看是不能通过指定列的索引值[0,1,2...]，而只能是行、列标签名来重排和添加新行、列数据

3. **fill_value**:
   - 在重排过程中，**如果有引入新的标签**，可以通过`fill_value`来设置一个**默认填充值**，而不是填充`NaN`。

4. **method**:
   - 用于填充`NaN`值的方法。例如，`ffill`表示前向填充，即使用前一个非`NaN`值来填充`NaN`。`bfill`是后向填充。

5. **limit**:
   - 当使用填充方法时，`limit`参数用来限制填充的数量。

6. **copy**:
   - 如果为`True`，即使新索引是等价的，也会返回一个新的对象。如果为`False`，并且没有必要更改数据，就不会返回一个新对象。

7. **level**:
   - 如果DataFrame有多级索引（MultiIndex），可以指定重排的级别。


在这个例子中，`df_reindexed` 的索引被重新排列成了 `['c', 'b', 'a', 'd']`，并且引入了一个新的索引 `'d'`，它在原始DataFrame中不存在，所以它的值被填充为`fill_value`指定的0。同样地，列也被重排为 `['C', 'A', 'B', 'D']`，并且引入了新列 `'D'`，其值同样被填充为0。

### （重要）参数详解

- `index`：该参数用于指定要保留的行索引。如果原始数据中**没有指定的行索引，则在结果中创建一个新行**。如果指定索引，则删除未包含在索引中的所有行。如果指定的索引存在于原始数据的行索引中，则保留该索引对应的原始数据行。例如：

  ```python
  import pandas as pd
  
  df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
  
  # 重新构建索引
  new_index = ['x', 'z', 'w']
  new_df = df.reindex(index=new_index)
  print(new_df)
  ```

  此处 `new_df` 的输出为：

  ```python
       a    b
  x  1.0  4.0
  z  3.0  6.0
  w  NaN  NaN
  ```

- `columns`：该参数用于指定要保留的列名（注意这里是列名，不是列索引）。如果原始数据中**没有指定的列名，则在结果中创建一个新列**。如果指定列名，则删除未包含在列名中的所有列。如果指定的列名存在于原始数据的列名中，则保留该列名对应的原始数据列。例如：

  ```python
  import pandas as pd
  
  df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
  
  # 重新构建列索引
  new_columns = ['b', 'a', 'c']
  new_df = df.reindex(columns=new_columns)
  print(new_df)
  ```

  此处 `new_df` 的输出为：

  ```python
     b  a   c
  0  4  1 NaN
  1  5  2 NaN
  2  6  3 NaN
  ```

- `fill_value`：该参数用于在重新索引过程中向新索引中添加缺失值。例如：

  ```python
  import pandas as pd
  
  s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
  
  # 重新构建索引，并添加缺失值
  new_index = ['a', 'b', 'd']
  new_s = s.reindex(index=new_index, fill_value=0)
  print(new_s)
  ```

  此处 `new_s` 的输出为：

  ```python
  a    1
  b    2
  d    0
  dtype: int64
  ```

- `method`：该参数用于指定当重新索引过程中添加新索引时使用的插值方法，它有以下四个可选值：

  - `'pad'` 或 `'ffill'`：使用前向填充方式。即将之前的值向下填充。
  - `'backfill'` 或 `'bfill'`：使用后向填充方式。即将之后的值向上填充。
  - `'nearest'`：使用最近邻插值方式，选择靠近索引的值填充。
  - `'None'` 或 `np.nan`：不做填充。默认方式为 None。

  例如：

  ```python
  import pandas as pd
  
  s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
  
  # 重新构建索引，并使用'nearest'插值方式填充缺失值
  new_index = ['a', 'b', 'd']
  new_s = s.reindex(index=new_index, method='nearest')
  print(new_s)
  ```

  此处 `new_s` 的输出为：

  ```python
  a    1
  b    2
  d    3
  dtype: int64
  ```

以上就是 `reindex()` 函数可选参数的介绍和示例，您可以根据实际情况选择合适的参数进行重新索引操作。