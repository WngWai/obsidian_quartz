`gghalves` 是一个 R 语言的包，专为 `ggplot2` 提供**半面图**（half-plots）的功能。这种图形非常适合于展示分布、比较组间差异或者关系。`gghalves` 的设计目的是为了让用户能够轻松地创建这些半面图，以增强数据可视化的表达力。

相互关联的图标合在一起！

![[Pasted image 20240303161118.png]]

`gghalves` 主要提供了一系列以 `geom_half_*` 形式命名的函数，这些函数基本上是 `ggplot2` 中相应 `geom_*` 函数的“半面”版本。下面将按功能对这些主要函数进行分类介绍：

### 分布展示

- **`geom_half_boxplot()`**: 创建半面箱形图，适用于展示数据分布的范围、中位数等统计特征。
- **`geom_half_violin()`**: 绘制半面小提琴图，用于展示数据的分布密度，比箱形图提供更多分布形状的信息。
- **`geom_half_dotplot()`**: 生成半面点阵图，用于表示数据分布的离散特征，特别是小样本数据集。

### 关系展示

- **`geom_half_point()`**: 绘制半面散点图，常用于展示两个变量之间的关系。
- **`geom_half_text()`**: 添加半面文本标签，通常与其他几何对象结合使用，用于显示数据点的具体值或其他信息。

### 比较展示

- **`geom_half_dumbbell()`**: 生成半面哑铃图，非常适用于比较两个分组或条件下的数值差异。

### 使用实例

下面是一个使用 `gghalves` 和 `ggplot2` 创建半面箱形图的简单例子：

```R
# 安装和加载 gghalves 和 ggplot2
if (!requireNamespace("gghalves", quietly = TRUE)) install.packages("gghalves")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
library(gghalves)
library(ggplot2)

# 创建示例数据
data(mpg, package = "ggplot2")

# 绘制半面箱形图
ggplot(mpg, aes(class, hwy)) +
  geom_half_boxplot() +
  coord_flip() # 翻转坐标轴以更好地展示水平箱形图
```

请注意，`gghalves` 的开发和更新可能不如 `ggplot2` 那样频繁，因此在使用时，请确保检查最新的文档和兼容性信息。

总的来说，`gghalves` 通过提供一系列“半面”几何对象，大大丰富了 `ggplot2` 的可视化工具箱，使其能够更加灵活地呈现数据的分布、关系和比较信息。