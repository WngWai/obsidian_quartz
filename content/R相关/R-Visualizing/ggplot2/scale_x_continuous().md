是 ggplot2 包中用于**调整 x 轴连续变量**的函数。它可以修改 x 轴的**显示范围、标签、刻度和其他属性**，以便更好地呈现数据。

```R
scale_x_continuous()
```

- `name`：指定 x 轴的名称或标题。

- `breaks`：指定 x 轴刻度的位置。可以使用**数字向量**来指定刻度的位置。

breaks = data_q1$air_date。x能等距隔开
![Pasted image 20231226165543](Pasted%20image%2020231226165543.png)


- `labels`：指定 x 轴**刻度标签**的**显示文本**。可以使用字符向量来指定刻度标签的文本。

	刻度标签向量的长度要跟之前的break参数向量的长度一致！
	可以理解为刻度**显示的别名**，如果2011-08-22的形式不合适，可以设置别名。
	labels=NULL，刻度标签不显示！

- date_lables
- date_breaks

- `limits`：指定 **x 轴显示的范围**。可以使用数字向量来指定范围的最小值和最大值。

- `expand`：指定 x 轴显示范围的扩展因子。可以使用数字向量 `[a, b]` 来设置扩展因子，其中 `a` 控制范围的下限扩展，`b` 控制范围的上限扩展。

- `trans`：指定 x 轴坐标轴的变换函数。可以使用函数如 `log10`、`sqrt` 等来对坐标轴进行变换。

下面是一个示例，展示如何使用 `scale_x_continuous()` 函数调整 x 轴的属性：
```R
library(ggplot2)

# 创建示例数据
df <- data.frame(
  x = c(1, 2, 3, 4, 5),
  y = c(2, 4, 6, 8, 10)
)

# 绘制散点图，并调整 x 轴的属性
ggplot(df, aes(x, y)) +
  geom_point() +
  scale_x_continuous(
    name = "X轴",
    breaks = c(1, 3, 5),
    labels = c("一", "三", "五"),
    limits = c(0, 6),
    expand = c(0.05, 0),
    trans = "log10"
  )
```

在上述示例中，我们创建了一个包含 x 和 y 值的数据框 `df`，然后使用 `ggplot()` 函数创建绘图对象，并使用 `geom_point()` 添加散点图。通过 `scale_x_continuous()` 函数，我们调整了 x 轴的多个属性，包括名称、刻度位置、刻度标签、显示范围、扩展因子和变换函数。

你可以根据自己的数据和需求，通过调整这些参数来定制和美化 x 轴。此外，`scale_x_continuous()` 还有其他参数可以使用，例如 `limits`（限制范围的自动计算）和 `breaks`（自定义刻度位置），你可以根据需要查阅 ggplot2 的文档来进一步了解。

### 设置直方图初始值
[scale_x_continuous()](.md) 限制了坐标轴的范围，但分组还是默认分组
```R
# input data
df_ppg <- read.csv("./data/NBAPlayerPts.csv")

# 
ggplot(df_ppg) +
  geom_histogram(aes(x=PPG)) +
  scale_x_continuous(limits = c(10, 30))
```
![Pasted image 20231108201222](Pasted%20image%2020231108201222.png)


把x轴的坐标和geom_histogram()中的组距、组数限制下，发现默认分组是11-13，而非10-12！
```R
# input data
df_ppg <- read.csv("./data/NBAPlayerPts.csv")

# 
ggplot(df_ppg) +
  geom_histogram(aes(x = PPG), binwidth = 2, bins = 15) +
  scale_x_continuous(breaks = seq(10,30,2), limits = c(10, 30)) +
  scale_y_continuous(breaks = seq(1,18))
```

![Pasted image 20231108224752](Pasted%20image%2020231108224752.png)

### date_labels 和date_breaks参数
在R语言中，`scale_x_continuous()`函数通常用于调整连续型数值轴（如X轴）的刻度。`date_labels`和`date_breaks`，它们是`ggplot2`包中专门用于处理日期和时间刻度的函数。
#### date_labels

`date_labels`参数用于指定日期或时间轴上标签的显示格式。此参数通过传入一个格式化字符串来控制日期时间的显示方式，该字符串遵循R语言中的`strptime()`和`format()`函数的格式规则。

例如：
- `"%Y"` 会显示为四位数的年份，如 "2023"。
- `"%b %d"` 会显示为缩写的月份和日期，如 "Jan 01"。
- `"%m/%d/%Y %H:%M"` 会显示完整的日期和小时分钟，如 "01/01/2023 15:00"。

#### date_breaks

`date_breaks`参数用于指定日期或时间轴上刻度的间隔。这个参数可以接受一个以时间单位表示的字符串，告诉ggplot2如何放置刻度。

例如：
- `"1 month"` 或 `"1m"` 会每个月放一个刻度。
- `"2 weeks"` 或 `"2w"` 会每两周放一个刻度。
- `"1 year"` 或 `"1y"` 会每年放一个刻度。

示例代码
以下是一个简单的例子展示了如何使用这些参数：

```R
library(ggplot2)

# 假设有以下包含日期的数据框
df <- data.frame(
  date = seq(as.Date("2020-01-01"), as.Date("2020-12-31"), by="month"),
  value = rnorm(12)
)

# 绘制图表，并设置日期轴的格式和间隔
ggplot(df, aes(x = date, y = value)) +
  geom_line() +
  scale_x_date(date_labels = "%b %d", date_breaks = "1 month") +
  theme_minimal()
```

在这个例子中，`scale_x_date()`用于将X轴设置为按月显示（每个月一个刻度），并且每个刻度的标签格式为缩写的月份和日期。