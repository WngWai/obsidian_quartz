函数是 `ggplot2` 包中用于**修改图形标签**的函数。它可以用来修改图形的标题、坐标轴标签和图例标签等。

以后统一在这设置标签！

```R
labs()
```
- `title`：用于修改**图形的标题**。
- `subtitle`：用于修改图形的**副标题**。
- `x`：用于修改 x 轴的标签。
x='x轴标题'
x = quote(sum(x[i]^2,i== 1,n))
y = quote(alpha + beta + frac(delta, theta))
[[quote()]]直接引入数据公式！这个公式是latex写法吧？


- `y`：用于修改 y 轴的标签。
- `caption`：用于修改**图形的注释**。

![[Pasted image 20240310141432.png]]


- quote 编写公式
labs(
x = quote(sum(x[i] ^ 2, i == 1, n)),
y = quote(alpha + beta + frac(delta, theta))
)

通过指定对应参数的值，可以对图形的标签进行修改。可以使用字符向量或表达式来设置标签的内容，也可以通过其他函数（如 `expression()`、`bquote()` 等）来实现更高级的标签表达式。

以下是一个简单的示例，演示如何使用 `labs()` 函数修改图形的标签：

```R
library(ggplot2)

# 示例数据集
df <- data.frame(x = 1:5, y = c(2, 4, 6, 8, 10))

# 创建绘图对象，定义映射关系
p <- ggplot(data = df, mapping = aes(x = x, y = y))

# 添加点图层
p <- p + geom_point()

# 修改图形标签
p <- p + labs(
  title = "Scatter Plot",
  subtitle = "Example",
  x = "X Axis",
  y = "Y Axis",
  caption = "DataSource: lj"
)

# 显示图形
print(p)
```
![Pasted image 20231018163031](Pasted%20image%2020231018163031.png)
在上述示例中，我们首先创建了一个数据集 `df`，然后使用 `ggplot()` 函数创建了一个基本的绘图对象 `p`，并在 `aes()` 函数中定义了 x 和 y 列与图形的映射关系。接着，通过 `+` 运算符和 `geom_point()` 函数，添加了一个点图层。最后，使用 `labs()` 函数修改了图形的标签，包括**标题、副标题、坐标轴标签和图形注释**。

通过调整 `labs()` 函数的参数，可以自定义图形的标签内容，以适应特定的需求。可以使用不同的字符、表达式或其他函数来实现丰富的标签表达式。