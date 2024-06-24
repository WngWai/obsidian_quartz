ggpairs(data)

```R
library(GGally)

note<- as_tibble(mclust::banknote)
ggpairs(note,aes(col = Status)) # 根据Status分配颜色
```

![[Pasted image 20240331162533.png]]

`ggpairs()`函数位于`GGally`包中，基于 `ggplot2` 包构建。该函数用于创建一个绘制多个变量之间关系的图表矩阵，可以用于探索**变量之间的相关性和分布情况**。




**函数定义**：
```R
ggpairs(data, mapping = aes(), columns = NULL, lower = NULL, diag = NULL, upper = NULL,
        title = NULL, ggplot2::theme = ggplot2::theme_bw(), progress = "none", ...)
```

**参数介绍**：
- `data`：数据框或数据集。
- `mapping=aes()`：变量之间的映射关系，可用于设置颜色、形状等。

详看aes()

- `columns`：选择要包含在图表中的列名。

- `lower`：在下三角部分绘制的图表类型或函数。
- `diag`：对角线上绘制的图表类型或函数。
- `upper`：在上三角部分绘制的图表类型或函数。

- `title`：图表的标题。
- `ggplot2::theme`：图表的主题。

- `progress`：可选的**进度条**选项，用于显示绘图进度。默认**True**显示

- `...`：其他传递给底层图表函数的参数。

**应用举例**：
以下是一个简单的应用示例，展示如何使用`ggpairs()`函数：

```R
library(GGally)

# 加载数据集
data(iris)

# 创建图表矩阵
ggpairs(iris[, 1:4], title = "Iris Data Set")
```

在上述示例中，我们首先加载了`GGally`包，并使用`data(iris)`加载了经典的鸢尾花数据集。

然后，我们使用`ggpairs()`函数创建了一个图表矩阵，传入了数据集`iris`的前四列作为变量。我们还设置了标题为"Iris Data Set"。

运行代码后，将绘制一个包含多个变量之间关系的图表矩阵，每个格子中展示相关性图、散点图或直方图等。

这是`ggpairs()`函数的简单应用示例。您可以根据需要调整参数来自定义图表矩阵的外观和显示方式。