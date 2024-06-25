
```python
pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
```

参数说明：
`arg`：需要转换成datetime类型的参数，可以是一个**字符串、列表、数组、Series或者DataFrame**；

`errors`：当转换失败时，如何处理错误，默认值为**raise**，即抛出异常，还可以选择**coerce**，即将错误值转换为**NaT**（not a time）；

- `dayfirst`：如果输入的字符串中，日期在月份之前，那么需要设置`dayfirst=True`，否则默认为`False`；
- `yearfirst`：如果输入的字符串中，年份在月份之前，那么需要设置`yearfirst=True`，否则默认为`False`；
- `utc`：如果输入的字符串中包含时区信息，那么需要设置`utc=True`，否则默认为`None`；
- `box`：如果输入的参数只有一个元素，那么默认情况下，转换后的结果是一个Timestamp对象，如果需要将其转换为标量，那么需要将`box=False`；

`format`：**日期时间字符串的格式代码**。如果输入的字符串的格式不是标准的ISO8601格式，那么需要设置`format`参数，用来指定字符串的格式；

- `exact`：如果输入的字符串中，存在非法的日期或者时间，那么默认情况下会抛出异常，如果需要忽略非法的日期或者时间，那么需要将`exact=False`；
- `unit`：可以指定时间戳的单位，如`'s'`表示秒，`'ms'`表示毫秒，`'us'`表示微秒，`'ns'`表示纳秒；
- `infer_datetime_format`：如果输入的字符串的格式是标准的ISO8601格式，那么默认情况下会快速解析，如果需要精确解析，需要将`infer_datetime_format=True`；
- `origin`：如果输入的时间戳是相对于某个时间点的，那么需要设置`origin`参数，比如Unix时间戳的起点是1970年1月1日；
- `cache`：如果需要多次转换同一个字符串，那么可以将`cache=True`，以提高性能。


下面是一些示例：

```python
import pandas as pd

# 将字符串转换为datetime
date_str = '2023-06-02'
date = pd.to_datetime(date_str)
print(date)

# 将多个字符串转换为datetime
date_strs = ['2023-06-01', '2023-06-02', '2023-06-03']
dates = pd.to_datetime(date_strs)
print(dates)

# 将Series对象转换为datetime
date_series = pd.Series(date_strs)
dates = pd.to_datetime(date_series)
print(dates)

# 将时间戳转换为datetime
timestamp = 1622601600
date = pd.to_datetime(timestamp, unit='s')
print(date)
```


#### 遇到时间超出DT范围的需要加errors处理错误
上面有相关参数介绍
```python
df_data['交易时间'] = pd.to_datetime(df_data['交易时间'], errors='coerce')
```

`to_datetime()`函数将输入转换为`datetime`类型的对象。`errors`是可选参数，用于控制在转换失败时的行为。 `errors`接受三个值：

- `raise`：如果输入**无法解析为日期时间**，则引发`TypeError`程序**报错**。这是**默认**行为。
- `ignore`：当无法解析输入时，返回**原始输入**。
- `coerce`：对于无法解析的输入，例如时间超出DT范围等，返回`NaT`，即`datetime`类型的“Not a Time”。

下面是一些示例：

```python
import pandas as pd

# 时间字符串和时间戳列表
date_strings = ['2023-01-01 00:00:00', '2023-01-02 00:00:00', '2023-01-03 00:00:00', '2023-01-04 00:00:00', '2023-01-05 00:00:00', 'error']
timestamps = [16801, 16802, 16803, 16804, 16805, 'error']

# 替代dtype
date_datetime = pd.to_datetime(date_strings, errors='coerce', infer_datetime_format=True)
date_timestamp = pd.to_datetime(timestamps, errors='coerce', unit='s')

print("date_strings to datetime: \n", date_datetime)
print("\n")
print("timestamps to datetime: \n", date_timestamp)
```

输出结果如下：

```python
date_strings to datetime: 
 DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', 'NaT'], dtype='datetime64[ns]', freq=None)


timestamps to datetime: 
 DatetimeIndex(['1970-01-01 00:00:01', '1970-01-01 00:00:02','1970-01-01 00:00:03', '1970-01-01 00:00:04', '1970-01-01 00:00:05', 'NaT'], dtype='datetime64[ns]', freq=None)
```

如你，输入字符串列表的最后一个元素无法转换为`datetime`对象，因此返回值为`NaT`。 对于时间戳列表，**时间戳的值太大**，因此返回值为`NaT`。

#### 数据类型转换
字符串转换为pandas datetime
通过to_datetime函数可以把**字符串**转换为pandas datetime

```python
    df = pd.DataFrame({'date': ['2011-04-24 01:30:00.000']})
    df['date'] = pd.to_datetime(df['date'])
```

打印结果

```
0   2011-04-24 01:30:00
Name: date, dtype: datetime64[ns]
```

如果字符串格式不正规，可以通过format转换，[参考](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)

```python
pd.to_datetime("20110424 01:30:00.000", format='%Y%m%d %H:%M:%S.%f')
```

#### [时间戳](https://so.csdn.net/so/search?q=%E6%97%B6%E9%97%B4%E6%88%B3&spm=1001.2101.3001.7020)转换为pandas datetime

to_datetime 如果传入的是10位时间戳，unit设置为秒，可以转换为datetime

```python
pd.to_datetime(1303608600, unit='s')
```

打印结果

```
2011-04-24 01:30:00
```

#### pandas datetime转换为时间戳

astype(‘int64’)//1e9 这种方式效率比较高

```python
    df = pd.DataFrame({'date': ['2011-04-24 01:30:00.000']})
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype('int64')//1e9
```

打印结果

```
0    1.303609e+09
Name: date, dtype: float64
```


### pandas datetime的比较
都是datetime的形式，如2023-01-02 00:00:00这种形式，就可以直接进行比较。交易先转变化字符串形式，注意消除空格。再转变为datetime数据类型，进行时间段的比较。

#### format的详细理解
对字符串指定规定格式进行时间转换。
`format` 参数用于指定日期时间字符串的格式代码，以确保解析的一致性和准确性。格式代码使用类似于 `strftime()` 函数的格式规范。下面是一些常见的日期时间格式代码及其解释：

- **%Y**: 四位数的年份（例如：2023）
- **%y**: 两位数的年份（例如：23）
- **%m**: 两位数的月份（01 到 12）
- **%d**: 两位数的日期（01 到 31）
- **%H**: 24 小时制的小时数（00 到 23）
- **%I**: 12 小时制的小时数（01 到 12）
- **%M**: 两位数的分钟数（00 到 59）
- **%S**: 两位数的秒数（00 到 59）
- **%f**: 微秒（六位数，范围从 000000 到 999999）
- **%p**: AM/PM 标记（仅适用于 12 小时制）
- **%z**: UTC 偏移量（例如：+0800）
- **%Z**: 时区名称（例如：CST）
- **%j**: 年份中的天数（001 到 366）
- **%U**: 年份中的周数（00 到 53，星期日为一周的第一天）
- **%W**: 年份中的周数（00 到 53，星期一为一周的第一天）
- **%c**: 日期时间的适当字符串表示（例如：Mon Sep 30 07:06:05 2023）
- **%x**: 适当的日期字符串表示（例如：09/30/23）
- **%X**: 适当的时间字符串表示（例如：07:06:05）

这些是常见的一些格式代码示例，您可以根据需要使用不同的格式代码来解析您的日期时间字符串。根据特定日期时间字符串的格式，您可以使用适当的格式代码来构建 `format` 参数的字符串。例如，如果日期时间字符串为 **"2023-08-28 12:30:00"**，则可以使用 `format='%Y-%m-%d %H:%M:%S'` 来指定该格式。

请注意，格式代码区分大小写，因此 `%Y` 表示四位数的年份，而 `%y` 表示两位数的年份。确保使用正确的格式代码以匹配日期时间字符串的实际格式。