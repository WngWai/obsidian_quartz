 是 ggplot2 包中用于**手动设置颜色映射**的函数。它允许你自定义图形中的颜色，为离散变量或分组变量指定特定的颜色值。

```R
scale_color_manual(values, guide = "legend", aesthetics = "colour")
```

- `values`：一个包含颜色的向量，用于手动设置颜色映射。可以是具体的颜色名称、十六进制颜色代码或 R 中已定义的颜色向量。

![Pasted image 20231122203026](Pasted%20image%2020231122203026.png)

- `labels`：指定每个颜色对应的**标签名**，用于**在图例中显示**。这个对应顺序应该可以指定

- `breaks`：指定要显示的颜色刻度的位置。

- `guide`：指定**图例的显示方式**。常用的选项包括 "legend"（默认，**显示图例**）和 "none"（不显示图例）。

- **`aesthetics`：** 指定要调整的颜色变量，通常为 "colour"。

下面是一个示例，展示如何使用 `scale_color_manual()` 函数手动设置颜色映射：
```R
library(ggplot2)

# 创建示例数据框
data <- data.frame(x = 1:5, y = 1:5, group = c("A", "A", "B", "B", "C"))

# 创建散点图
ggplot(data, aes(x = x, y = y, color = group)) +
  geom_point(size = 3) +
  scale_color_manual(
    values = c("A" = "red", "B" = "blue", "C" = "green"),
    labels = c("Group A", "Group B", "Group C"),
    breaks = c("A", "B", "C"),
    guide = "legend"
  ) +
  labs(title = "Scatter Plot", x = "X-axis", y = "Y-axis")
```

在上述示例中，我们首先创建了一个名为 `data` 的数据框，其中包含了三个变量 `x`、`y` 和 `group`。然后，使用 `ggplot()` 函数创建一个基本的绘图对象，并使用 `geom_point()` 函数添加散点图的图层。通过 `aes()` 函数将 `group` 变量映射到颜色（color）属性。

通过 `scale_color_manual()` 函数手动设置颜色映射。通过 `values` 参数指定了每个组别（A、B、C）对应的颜色（"red"、"blue"、"green"）。通过 `labels` 参数指定了每个颜色对应的标签（"Group A"、"Group B"、"Group C"），用于在图例中显示。通过 `breaks` 参数指定了要显示的颜色刻度的位置（"A"、"B"、"C"）。通过 `guide` 参数设置了图例的显示方式为 "legend"，即显示图例。


![[Pasted image 20240322151435.png]]

```python
presidential %>%
mutate(id=33+ row_number()) %>%
ggplot(aes(start, id,colour = party))+
	geom_point()+
	geom_segment(aes(xend = end,yend = id))+
	scale_colour_manual(values = c(Republican="red",Democratic ="blue"))
```