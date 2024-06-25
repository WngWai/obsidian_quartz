在 Python 中，`dt.strftime()` 方法是 `datetime` 类的一个方法，用于将 `datetime` 对象格式化为字符串。这个方法**使用一个格式字符串来指定输出的格式**。
### 定义
`dt.strftime(format)` 方法接受一个字符串参数 `format`，该字符串定义了如何将 `datetime` 对象转换为字符串。

```python
from datetime import datetime
# 假设 dt 是一个 datetime 对象
dt = datetime(2023, 11, 2, 15, 30, 45)
# 使用 strftime 方法格式化 dt 为字符串
formatted_date_time = dt.strftime('%Y-%m-%d %H:%M:%S')
```
### 常用格式字符串
以下是 `strftime` 方法中常用的格式字符串：
- `%Y`：四位数的年份（例如，2023）
- `%m`：月份（01-12）
- `%d`：月中的一天（01-31）
- `%H`：小时（24 小时制，00-23）
- `%M`：分钟（00-59）
- `%S`：秒（00-59）
- `%A`：星期几的全称（例如，Monday）
- `%a`：星期几的简称（例如，Mon）
- `%B`：月份的全称（例如，November）
- `%b`：月份的简称（例如，Nov）
- `%I`：小时（12 小时制，01-12）
- `%p`：上午或下午的指示符（AM 或 PM）
### 使用举例
```python
from datetime import datetime
# 创建一个 datetime 对象
dt = datetime(2023, 11, 2, 15, 30, 45)
# 格式化为 YYYY-MM-DD HH:MM:SS 格式
print(dt.strftime('%Y-%m-%d %H:%M:%S'))  # 输出: 2023-11-02 15:30:45
# 格式化为 MM/DD/YYYY HH:MM:SS AM/PM 格式
print(dt.strftime('%m/%d/%Y %I:%M:%S %p'))  # 输出: 11/02/2023 03:30:45 PM
# 格式化为星期几的简称和月份的简称
print(dt.strftime('%a, %b'))  # 输出: Thu, Nov
```
在这个示例中，我们创建了一个 `datetime` 对象，并使用 `strftime` 方法将其格式化为不同的字符串格式。通过使用不同的格式字符串，我们可以自定义日期和时间的输出格式。
