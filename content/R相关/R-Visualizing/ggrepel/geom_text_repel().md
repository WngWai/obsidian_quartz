`ggrepel` 包中的 `geom_text_repel()` 函数是 `ggplot2` 包的扩展，用于创建带有避免重叠的文本标签的图。这个函数特别有用，当你在图表中有很多标签时，而这些标签如果重叠会使得图表难以阅读。

![[Pasted image 20240321151950.png]]


```R
geom_text_repel(mapping = NULL, data = NULL, ..., parse = FALSE,
                nudge_x = 0, nudge_y = 0, nudge_units = "pt",
                direction = "both", force = 1, 
                box.padding = 0.25, point.padding = 0.5,
                segment.color = NULL, segment.size = 0.5,
                arrow = NULL, na.rm = FALSE, show.legend = NA, 
                inherit.aes = TRUE)
```

### 主要参数

- `mapping`: aesthetic 映射，例如 `aes(x, y, label)`.
- `data`: 数据框，包含要绘制的数据。
- `parse`: 如果为 TRUE，则标签文本将被解析。
- `nudge_x`, `nudge_y`: 标签在 x 或 y 方向上的微调。
- `direction`: 排斥方向（"both", "x", "y"）。
- `force`: 排斥力大小。
- `box.padding`, `point.padding`: 分别是标签周围的填充和标签与其指向点的距离。
- `segment.color`, `segment.size`: 连接标签和其指向点的线段颜色和大小。
- `arrow`: 如需在标签和指向点之间添加箭头，指定箭头样式。

### 应用举例
其中geom_point添加了个圈！使得标注更清晰！
```r
ggplot(mpg,aes(displ,hwy))+

geom_point(aes(colour = class))+

geom_point(size=3,shape=1,data = best_in_class)+

ggrepel::geom_label_repel(aes(lahel = model),data = best_in_class)
```

以上代码创建了一个散点图，并使用 `geom_text_repel()` 函数添加了不重叠的文本标签。每个标签与其数据点通过灰色线段连接，同时保持了一定的距离以提高可读性。