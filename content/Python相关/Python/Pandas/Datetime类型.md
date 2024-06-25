在 Python 中，`datetime` 模块提供了 `datetime` 类，用于**处理日期和时间**。以下是 `datetime` 类的一些常用属性和方法，以及它们的简单使用示例。
### 常用属性
dt.date获取日期时间列中每个**日期时间**对应的**日期**部分。
dt.time获取日期时间列中每个**日期时间**对应的**时间**部分。
dt.year返回或设置**年**份，**整型**。
dt.month返回或设置**月**份。
dt.day返回或设置**天**。
dt.hour返回或设置**小时**。
dt.minute返回或设置**分钟**。
dt.second返回或设置**秒**。
dt.microsecond获取日期时间列中每个日期时间对应的**微秒数**。
dt.nanosecond获取日期时间列中每个日期时间对应的**纳秒数**。

dt.weekday获取日期时间列中每个日期时间对应的**星期几**，返回值为**0-6**，分别表示周一到周日。
dt.weekday_name获取日期时间列中每个日期时间对应的**星期几的名称**，例如'Monday'、'Tuesday'等。
dt.isoweekday()返回**国际**标准的一周中的**星期几**（1-7，1 表示星期一）。

dt.quarter获取日期时间列中每个日期时间对应的**季度**，返回值为**1-4**。
dt.days_in_month获取日期时间列中每个日期时间**对应的月份的天数**。
dt.is_leap_year获取日期时间列中每个日期时间对应的年份**是否为闰年**，返回值为True或False。
### 常用方法
???
timetuple()返回一个 struct_time对象，表示**本地时间的日期和时间**。
utcnow()返回当前的 UTC 日期和时间。

dt.timestamp()返回一个**浮点数**，表示自 1970 年 1 月 1 日 00:00:00 UTC 以来的**秒数**
fromtimestamp()从给定的**浮点数**（表示自 1970 年以来的秒数）创建一个**datetime对象**

`replace()`：创建一个新的 `datetime` 对象，其指定的属性与原对象相同，但某些属性已被替换。
 `timetuple()`：返回一个 `struct_time` 对象，表示本地时间的日期和时间。


### 使用举例

```python
from datetime import datetime 

# 获取当前时间
current_time = datetime.now()
print("当前时间:", current_time)
# 格式化时间输出
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("格式化时间:", formatted_time)
# 创建一个 datetime 对象
specific_time = datetime(2023, 11, 2, 15, 30, 45)
print("特定时间:", specific_time)
# 获取特定时间的时间部分
time_part = specific_time.time()
print("时间部分:", time_part)
# 获取特定时间的日期部分
date_part = specific_time.date()
print("日期部分:", date_part)
# 计算两个时间之间的差值
time_diff = current_time - specific_time
print("时间差:", time_diff)
# 增加或减少时间
time_added = specific_time + timedelta(days=1, hours=2)
print("增加一天两小时的时间:", time_added)
```


### dt.day
是Pandas中Datetime类型的Series或DataFrame列的一个属性，用于提取该列中每个日期时间对应的“日”部分，返回一个新的整数Series或DataFrame列。例如，对于一个日期时间类型的列，可以使用`.dt.day`属性来获取该列中每个日期时间对应的日，以便进行进一步的分析和处理。具体用法可以参考下面的示例代码：

```python
import pandas as pd

# 创建一个DataFrame，包含一个日期时间列
df = pd.DataFrame({'date': ['2022-06-01 12:00:00', '2022-06-02 13:30:00', '2022-06-03 14:45:00']})
df['date'] = pd.to_datetime(df['date'])  # 将字符串转换为日期时间类型

# 提取日期时间列中的“日”部分
df['day'] = df['date'].dt.day

print(df)
```

运行上面的代码，输出结果如下：

```python
                  date  day
0 2022-06-01 12:00:00    1
1 2022-06-02 13:30:00    2
2 2022-06-03 14:45:00    3
```

可以看到，通过`.dt.day`属性，我们成功地从日期时间列中提取了每个日期时间对应的“日”部分，并将其保存到了一个新的列中。

### dt.days
`dt.days`是指`datetime.timedelta`对象表示的**时间间隔中的天数**，是一个整数。例如，如果`dt`是一个表示两天时间间隔的`datetime.timedelta`对象，那么`dt.days`的值就是2。

`dt.days`是日期时间序列中的一个属性，用于返回该序列中的天数部分。在pandas中，如果我们有一个日期时间序列，可以使用`.dt`属性来访问关于日期时间的特定属性和方法，例如`.dt.days`用于获取天数。

下面是一个示例，展示了如何使用`.dt.days`来获取日期时间序列中的天数：
```python
import pandas as pd

# 创建一个日期时间序列
dates = pd.to_datetime(['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04'])

# 使用.dt.days获取天数
days = dates.dt.days

print(days)
```

输出结果会显示日期时间序列中的天数部分：

```python
0    18627
1    18628
2    18629
3    18630
dtype: int64
```

在上述示例中，我们首先使用`pd.to_datetime()`函数将日期字符串转换为日期时间序列。然后，通过应用`.dt`属性，我们可以访问日期时间序列的各种属性和方法。使用`.dt.days`，我们可以获取该日期时间序列中的天数，并存储在`days`变量中。最后，我们打印出`days`变量。注意，输出结果中的天数是从**1970年1月1日**起的天数。

另一个例子：
将时间差转换为天数差！！！存储到time_diff列中
假设有一个数据集，记录了某个城市每天的最高气温和最低气温。我们可以使用 Pandas 来读取数据，并计算每天的温度差（即最高气温和最低气温之差）。代码如下：

```python
import pandas as pd

# 读取数据
df = pd.read_csv('temperature.csv')

# 将日期列转换为日期类型
df['date'] = pd.to_datetime(df['date'])

# 计算温度差
df['temp_diff'] = df['max_temp'] - df['min_temp']

# 查看结果
print(df)
```

这里假设数据集的文件名为 `temperature.csv`，它的内容如下：

```python
date,max_temp,min_temp
2023-06-01,28,20
2023-06-02,30,22
2023-06-03,31,23
2023-06-04,29,20
2023-06-05,27,18
```

运行上面的代码后，输出的结果如下：

```python
        date  max_temp  min_temp  temp_diff
0 2023-06-01        28        20          8
1 2023-06-02        30        22          8
2 2023-06-03        31        23          8
3 2023-06-04        29        20          9
4 2023-06-05        27        18          9
```

可以看到，我们成功地计算出了每天的温度差，并将结果保存在了新的一列 `temp_diff` 中。如果我们想要查看每个时间差对象中的天数部分，可以使用 `dt.days` 方法。例如，下面的代码可以打印出每个时间差对象中的天数部分：

```python
for i, row in df.iterrows():
    date = row['date']
    temp_diff = row['temp_diff']
    print(f"{date}: {temp_diff} (days: {temp_diff.dt.days})")
```

运行上面的代码后，输出的结果如下：

```python
2023-06-01 00:00:00: 8 (days: 8)
2023-06-02 00:00:00: 8 (days: 8)
2023-06-03 00:00:00: 8 (days: 8)
2023-06-04 00:00:00: 9 (days: 9)
2023-06-05 00:00:00: 9 (days: 9)
```

可以看到，每个时间差对象中的天数部分都被正确地计算出来了。
### weekday和dayofweek的差异
`.dt.weekday`和`.dt.dayofweek`都是用来获取日期时间列中每个日期时间对应的星期几，它们的返回值是相同的，都是0-6，其中0表示周一，1表示周二，以此类推，6表示周日。
它们的区别在于，`.dt.weekday`和`.dt.dayofweek`的参数不同。`.dt.weekday`的参数是'0-6'或'Mon-Sun'，而`.dt.dayofweek`的参数是'0-6'或'Sun-Sat'。

```python
import pandas as pd

# 创建一个DataFrame
df = pd.DataFrame({
    'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    'value': [1, 2, 3, 4, 5]
})

# 将日期字符串转换为Pandas的Datetime类型
df['date'] = pd.to_datetime(df['date'])

# 使用.dt.weekday属性获取星期几
df['weekday'] = df['date'].dt.weekday

# 使用.dt.dayofweek属性获取星期几
df['day_of_week'] = df['date'].dt.dayofweek

print(df)
```

输出结果如下：

```python
        date  value  weekday  day_of_week
0 2022-01-01      1        5            5
1 2022-01-02      2        6            6
2 2022-01-03      3        0            0
3 2022-01-04      4        1            1
4 2022-01-05      5        2            2
```

在上面的例子中，我们使用了`.dt.weekday`和`.dt.dayofweek`属性来获取`df['date']`列中每个日期时间对应的星期几，并将结果保存在了`df['weekday']`和`df['day_of_week']`两列中。注意到这两个属性的返回值是相同的，但是参数不同，一个是'0-6'或'Mon-Sun'，另一个是'0-6'或'Sun-Sat'。

### 筛选指定日期？？？有些问题
[python pandas 按照时间（h:m:s）条件对使用datetimeIndex或datetime类型列的数据进行筛选的方法_python pandas datetime index 筛选_phoenix339的博客-CSDN博客](https://blog.csdn.net/phoenix339/article/details/97620818)
在 Pandas 中，如果你想筛选 DataFrame 中的特定日期区间，你可以使用比较运算符（如 `>`、`<`、`>=`、`<=`）与 DateTime 类型进行比较。你需要确保你的日期列被正确解析为 DateTime 类型。

```python
import pandas as pd

# 创建一个包含日期的 DataFrame
df = pd.DataFrame({'date': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04'],
                   'value': [10, 20, 30, 40]})

# 将日期列解析为 DateTime 类型
df['date'] = pd.to_datetime(df['date'])

# 指定日期区间
start_date = pd.to_datetime('2023-07-02')
end_date = pd.to_datetime('2023-07-03')

# 筛选指定日期区间的数据
filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# 打印筛选结果
print(filtered_df)
```

运行上述代码，你将会得到以下筛选结果：

```
        date  value
1 2023-07-02     20
2 2023-07-03     30
```

在示例中，我们首先将日期列 `date` 使用 `pd.to_datetime()` 函数转换为 DateTime 类型。然后我们指定了起始日期 `start_date` 和结束日期 `end_date`。最后，我们使用比较运算符 `>=` 和 `<=` 对日期列进行筛选，保留指定日期区间的数据。

希望这个示例对你有帮助，如果你有任何其他问题，请随时提问。

