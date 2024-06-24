它用于创建一个**计数图**，通过绘制具有计数加权的点来表示观测次数的多少。这在处理具有重复观测的数据集时非常有用，可以让我们直观地看到数据点的密度。

![[Pasted image 20240314110130.png]]


```R
geom_count(mapping = NULL, data = NULL, stat = "sum", position = "identity", ..., show.legend = NA, inherit.aes = TRUE)
```

- `mapping`: 设置数据的美学映射，通常在 `aes()` 函数中定义，比如 `aes(x = var1, y = var2)`。
- `data`: 指定使用的数据集。如果在 `ggplot()` 函数中已经指定，这里可以不再设置。
- `stat`: 用于计算图层统计变换，**默认为 sum**，不常改变。
- `position`: 调整位置，用于防止图形重叠。常见的有 `"identity"`, `"stack"`, `"dodge"` 等。

- `...`: 其他与图层美学有关的参数，如 `size`, `shape`, `color`, `fill` 等。

- `show.legend`: 是否在**图例**中显示该图层，默认为 `NA`，根据情况自动判断。
- `inherit.aes`: **是否继承全局美学映射**，默认为 `TRUE`。

### 应用举例

假设我们有一个包含两个变量 `x` 和 `y` 的数据框 `df`，其中某些组合的 `(x, y)` 值是重复的，我们想要可视化这种重复性的分布。

```R
# 加载必要的库
library(ggplot2)

# 创建一个示例数据框
set.seed(123)
df <- data.frame(
  x = sample(1:10, 100, replace = TRUE),
  y = sample(1:10, 100, replace = TRUE)
)

# 使用 geom_count 绘图
ggplot(df, aes(x = x, y = y)) +
  geom_count() +
  theme_minimal() +
  labs(title = "geom_count 示例",
       x = "X 轴",
       y = "Y 轴")
```

这个例子首先生成了一个包含 `x` 和 `y` 两个变量的数据框 `df`，其中 `x` 和 `y` 的值都是从 1 到 10 之间的随机整数，总共有 100 对值，因此一些 `(x, y)` 组合是重复的。接下来，使用 `geom_count()` 创建一个图，其中点的大小表示在数据集中 `(x, y)` 组合出现的频率。这对于观察哪些组合更常见特别有用。

`geom_count()` 适合处理和展示有重复观测值的数据点，通过调整点的大小来直观地显示观测次数的多少，从而使数据的分布情况一目了然。