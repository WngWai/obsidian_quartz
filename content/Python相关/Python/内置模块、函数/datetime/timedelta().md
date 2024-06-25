在 Python 中，`timedelta()` 函数用于创建一个表示**时间间隔的对象**。这个对象可以表示两个日期或时间之间的差异。
### 定义
`timedelta()` 函数可以接受任意数量的参数，这些参数表示时间间隔的各个部分。参数可以是以下之一：
- `days`：表示**天数**。
- `seconds`：表示**秒数**。
- `microseconds`：表示微秒数。
- `milliseconds`：表示毫秒数（这是 `seconds` 的一个快捷方式，因为 `seconds` 参数接受的值是毫秒数）。
- `minutes`：表示**分钟数**（这是 `seconds` 的一个快捷方式，因为 `seconds` 参数接受的值是分钟数乘以 60）。
- `hours`：表示**小时数**（这是 `seconds` 的一个快捷方式，因为 `seconds` 参数接受的值是小时数乘以 3600）。
### 常用属性
`timedelta` 对象具有以下常用属性：
1. `days`：返回时间间隔的天数部分。
2. `seconds`：返回时间间隔的秒数部分。
3. `microseconds`：返回时间间隔的微秒数部分。
4. `total_seconds()`：返回时间间隔的总秒数，包括天数、秒数和微秒数。
### 使用举例
```python
from datetime import timedelta, datetime
# 创建一个 timedelta 对象，表示 2 天 3 小时 45 分钟
time_delta = timedelta(days=2, hours=3, minutes=45)
# 创建一个当前时间的 datetime 对象
current_time = datetime.now()
# 创建一个未来的 datetime 对象
future_time = current_time + time_delta
# 创建一个过去的 datetime 对象
past_time = current_time - time_delta
# 输出当前时间、未来时间和过去时间
print("当前时间:", current_time)
print("未来时间:", future_time)
print("过去时间:", past_time)
# 输出时间间隔的天数、秒数和微秒数
print("天数:", time_delta.days)
print("秒数:", time_delta.seconds)
print("微秒数:", time_delta.microseconds)
# 输出时间间隔的总秒数
print("总秒数:", time_delta.total_seconds())
```
在这个示例中，我们创建了一个 `timedelta` 对象，表示两天三小时四十五分钟的时间间隔。然后，我们使用这个时间间隔来计算当前时间之后的未来时间和之前的过去时间。我们还展示了如何获取时间间隔的各个部分，包括总秒数。
