python中的time包是一个用于处理时间的标准库，它提供了获取系统时间并格式化输出的功能，以及提供系统级精确计时的功能，用于程序性能分析。

常用：









- 时间获取函数：用于获取当前时间或指定时间的不同表示形式，如时间戳，元组，字符串等。例如：
    * `time.time()`：返回当前时间的时间戳，即从1970年1月1日0时0分0秒（UTC）到现在的秒数，是一个浮点数。
    * `time.ctime([secs])`：将时间戳转换为一个字符串，表示本地时间，格式为'Wed Jun 9 04:26:40 1993'，如果不提供secs参数，则使用当前时间的时间戳。
    * `time.gmtime([secs])`：将时间戳转换为一个元组，表示UTC时间，元组中包含9个整数，分别是年，月，日，时，分，秒，一周中的第几天，一年中的第几天，夏令时标志。
    * `time.localtime([secs])`：将时间戳转换为一个元组，表示本地时间，元组中包含9个整数，分别是年，月，日，时，分，秒，一周中的第几天，一年中的第几天，夏令时标志。
    * `time.mktime(t)`：将一个元组，表示本地时间，转换为一个时间戳，t是一个包含9个整数的元组，与`time.localtime()`返回的元组格式相同。
    * `time.perf_counter()`：返回一个高精度的性能计数器，表示从某个固定的时间点开始的秒数，是一个浮点数，可以用于程序性能分析。
    * `time.process_time()`：返回一个进程时间，表示当前进程使用CPU的秒数，是一个浮点数，可以用于程序性能分析。
    * `time.sleep(secs)`：让当前线程暂停secs秒，secs是一个浮点数，可以有小数部分。

- 时间格式化函数：用于将时间戳，元组，字符串之间进行转换，或按照指定的格式输出时间。例如：
    * `time.asctime([t])`：将一个元组，表示本地时间，转换为一个字符串，格式为'Wed Jun 9 04:26:40 1993'，如果不提供t参数，则使用当前时间的元组。
    * `time.strftime(format[, t])`：将一个元组，表示本地时间，按照指定的格式转换为一个字符串，format是一个格式字符串，可以包含各种格式化代码，如%Y表示年份，%m表示月份，%d表示日期等，如果不提供t参数，则使用当前时间的元组。
    * `time.strptime(string[, format])`：将一个字符串，表示本地时间，按照指定的格式转换为一个元组，string是一个时间字符串，format是一个格式字符串，与`time.strftime()`的格式字符串相同，如果不提供format参数，则使用默认的格式'%a %b %d %H:%M:%S %Y'。
    * `time.ctime([secs])`：与`time.asctime(time.localtime(secs))`等价。
    * `time.asctime([t])`：与`time.strftime('%a %b %d %H:%M:%S %Y', t)`等价。

- 其他函数：用于获取或设置时钟信息，时区信息，夏令时信息等。例如：
    * `time.tzset()`：根据环境变量TZ重新初始化时区设置，影响`time.localtime()`，`time.gmtime()`，`time.ctime()`等函数的结果。
    * `time.tzname`：一个元组，包含两个字符串，分别表示标准时区名称和夏令时名称，如('CST', 'CDT')。
    * `time.timezone`：一个整数，表示标准时区与UTC时区的时间差，以秒为单位，如-21600表示UTC-6。
    * `time.altzone`：一个整数，表示夏令时与UTC时区的时间差，以秒为单位，如-18000表示UTC-5。
    * `time.daylight`：一个整数，表示当前地区是否实行夏令时，0表示否，非0表示是。
    * `time.get_clock_info(name)`：返回一个命名元组，包含指定时钟的信息，name是一个字符串，可以是'monotonic'，'perf_counter'，'process_time'，'time'等，命名元组中包含5个字段，分别是implementation，monotonic，adjustable，resolution，current。


