在Python的Pandas库中，`Timedelta` 类用于表示时间差。它是一个不可变的时间对象，用于表示两个时间点之间的差异。`Timedelta` 对象可以表示天、小时、分钟、秒和微秒。
### 定义和参数介绍：
`Timedelta` 类的基本定义如下：
```python
class pandas.Timedelta(object):
    ...
```
`Timedelta` 对象可以通过多种方式创建，包括使用参数来指定不同的时间单位。以下是创建 `Timedelta` 对象的一些方法：
1. **使用参数创建**：
   - `Timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0)`：允许你指定天、秒、微秒、毫秒、分钟和小时。
2. **使用字符串创建**：
   - `Timedelta(delta_str)`：允许你使用字符串来表示时间差，例如 `'1D'` 表示1天，`'1H30M'` 表示1小时30分钟。
3. **使用其他`Timedelta`对象创建**：
   - `Timedelta(other)`：允许你使用另一个 `Timedelta` 对象来创建一个新的 `Timedelta` 对象。
### 应用举例：
以下是一些使用 `Timedelta` 类的基本示例：
**创建 Timedelta 对象**：
```python
import pandas as pd
# 使用参数创建
td1 = pd.Timedelta(days=1, seconds=30)
# 使用字符串创建
td2 = pd.Timedelta(delta_str='1D 30S')
# 使用其他Timedelta对象创建
td3 = pd.Timedelta(td1)
```
在这个例子中，我们创建了三个 `Timedelta` 对象，分别使用参数、字符串和另一个 `Timedelta` 对象。

**基本操作**：
```python
# 加法操作
td4 = td1 + td2
# 减法操作
td5 = td1 - td2
# 乘法操作
td6 = td1 * 2
# 除法操作
td7 = td1 / 2
```
在这个例子中，我们展示了如何对 `Timedelta` 对象进行加法、减法、乘法和除法操作。

**比较操作**：
```python
# 比较操作
print(td1 > td2)  # 输出：False
print(td1 < td2)  # 输出：True
```
在这个例子中，我们展示了如何对 `Timedelta` 对象进行比较操作。
总之，`Timedelta` 类是Pandas库中用于表示时间差的重要类之一。通过使用不同的创建方法，你可以轻松地创建和操作 `Timedelta` 对象。
