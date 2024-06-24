`patchwork` 是一个灵活的 R 包，用于将多个 `ggplot2` **图表组合成一个整体图形**。它提供了一种简单直观的语法，通过操作符和函数来组织和定制图形的布局。以下是一些 `patchwork` 包中的主要函数，按照它们的功能进行分类：

有用R语言绘图的，哪个包你最常用？ - 简佐义的回答 - 知乎
https://www.zhihu.com/question/495598810/answer/2953886230

### 1. 组合图形

- `+`：将两个图形水平或垂直组合在一起。默认情况下，`patchwork` 会自动决定是水平还是垂直组合，但这可以通过显式设置来控制。
- `|`：将两个图形水平组合在一起。
- `/`：将两个图形垂直组合在一起。

嵌入图
inset_element

### 2. 定义图形布局

- `plot_layout()`：定义图形之间的布局，包括图形的行和列数，以及每个图形所占的空间大小。它允许创建更复杂和定制化的布局。
- `plot_spacer()`：在图形间插入空白区域，有助于调整图形之间的间距。

### 3. 控制图形属性

- `plot_annotation()`：添加整个组合图的标题、子标题、标签或脚注。
- `guide_area()`：在组合图中集中显示图例，有助于避免重复的图例占据空间。

### 4. 定制和调整

- `theme()`：调整组合图的整体主题，这一点上`patchwork`继承自`ggplot2`的功能，允许在整个组合图上应用统一的主题设置。
- `&`：用于将`ggplot2`的主题或其他设置应用到组合图上，确保这些设置能够覆盖组合图中的每个单独图形。

### 示例代码

```r
library(ggplot2)
library(patchwork)

# 创建两个ggplot2图形
p1 <- ggplot(mtcars) + geom_point(aes(mpg, disp)) + ggtitle("Scatter plot")
p2 <- ggplot(mtcars) + geom_boxplot(aes(gear, disp, group = gear)) + ggtitle("Box plot")

# 使用patchwork组合图形
combined_plot <- p1 / p2 + 
  plot_layout(guides = 'collect') + 
  plot_annotation(title = 'Combined Plot with patchwork')

# 显示组合图形
print(combined_plot)
```

在这个示例中，两个`ggplot2`图形通过`patchwork`的`/`操作符垂直组合在一起，同时通过`plot_layout()`和`plot_annotation()`函数来定制布局和添加标题。`guides = 'collect'`选项在`plot_layout()`函数中用来集中显示两个图形的图例。

`patchwork`包通过提供简洁而强大的语法，极大地增强了在R语言中组合和展示多个`ggplot2`图形的能力。