`geom_label_repel()` 是 R 语言 `ggrepel` 包中的一个函数，用于在 `ggplot2` 图形中添加带有背景框的文本标签，同时自动调整标签的位置以减少重叠。

![[Pasted image 20240321153050.png]]

theme(legend.position ="none") 不显示图例
![[Pasted image 20240321153439.png]]

```R
geom_label_repel(mapping = NULL, data = NULL, ..., parse = FALSE,
                 nudge_x = 0, nudge_y = 0, nudge_units = "pt",
                 direction = "both", force = 1, 
                 box.padding = 0.25, point.padding = 0.5,
                 segment.color = NULL, segment.size = 0.5,
                 arrow = NULL, na.rm = FALSE, show.legend = NA, 
                 inherit.aes = TRUE)
```

此函数的参数与 `geom_text_repel()` 几乎完全相同，例如 `mapping`, `data`, `parse`, `nudge_x`, `nudge_y`, `direction`, `force`, `box.padding`, `point.padding`, `segment.color`, `segment.size`, `arrow`, `na.rm`, `show.legend`, `inherit.aes` 等。


假设你有一个包含城市及其对应人口和幸福指数的数据框 `cities_df`，你想通过散点图展示各个城市的人口与幸福指数，并且在每个点旁边显示带背景框的城市名称，避免重叠。

```R
library(ggplot2)
library(ggrepel)

elass_avg <- mpg %>%
	group_by(class) %>%
	summarise(
		displ = median(displ),
		hwy = median(hwy)
	)
	
ggplot (mpg, aes(displ, hwy, colour = elass)) +
	ggrepel::geom_label_repel(aes(label = class),
	data = class_avg,
	size=6,
	label.size = 0,
	segment.color = NA
	 ) +
	geom_point()+
	theme(legend.position ="none")
```

这段代码首先加载必要的包，定义数据框 `cities_df`，然后使用 `ggplot` 绘制人口-幸福指数散点图，通过 `geom_label_repel()` 添加每个点的带背景框城市名称标签，自动调整位置以减少重叠。




