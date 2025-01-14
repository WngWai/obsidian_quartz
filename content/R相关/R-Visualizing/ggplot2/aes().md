aesthetic用于定义**数据变量与图形的映射关系**。它用于将数据的变量与图形的视觉属性之间建立对应关系，例如将数据的**某一列映射到 x 轴位置、将另一列映射到 y 轴位置、将第三列映射到颜色**等。

通过在 `ggplot()` 函数中使用 `aes()` 函数**来定义映射关系**，可以将数据的不同变量与图形的不同视觉属性关联起来。这样，在后续的绘图过程中，可以直接使用这些映射关系，使得图形能够根据数据的变化而自动调整属性，展示出更丰富的信息。

```R
aes(x = NULL, y = NULL, ..., color = NULL, fill = NULL, shape = NULL, size = NULL, alpha = NULL)
```

- `x`、`y`：指定数据变量与 x 轴和 y 轴的映射关系。可以是数据框中的列名、向量、公式等。

	`..density..`：
```R
ggplot(data = diamonds, mapping = aes(x = price, y = ..density..)) + geom_freqpoly(mapping = aes(colour = cut), binwidth = 500)
```
特殊的语法，用于指代由`geom_histogram`、`geom_density`、`geom_freqpoly`等几何对象计算得到的统计变换结果，直接在映射（`aes()`函数内）中引用计算后的值。
上面带代码中的..density..指每个bin的计数除以总计数乘以bin宽度，即密度。

- `...`：其他映射关系。

都可以按照另一个变量分组映射，但必须另一个变量是**字符型**！

![Pasted image 20231024195533](Pasted%20image%2020231024195533.png)

- `color`：指定数据变量与图形的**颜色**映射关系，（如"red"、"blue"），也可以使用十六进制颜色码（如"#FF0000"表示红色）。另一个变量**可以是`数值`**！

- `fill`：指定数据变量与图形的**填充颜色**映射关系，。

- `shape`：指定数据变量与图形的**形状**映射关系。控制点的形状，如圆形、方形、三角形等。

- `size`：指定数据变量与图形的**大小**映射关系，控制图形元素的大小，如点的大小或线的粗细。

- `stroke`：指定边框颜色的数据变量或常量值。适用于具有边框的图形，如点的边框颜色。

- `width`：指定宽度的数据变量或常量值。用于调整线的宽度。

- `alpha`：指定**透明**度（alpha）的数据变量或常量值。取值范围为0（完全透明）到1（完全不透明）之间。

- `linetype`：指定**线型的数据变量或常量值**。控制线条的类型，如实线、虚线、点线等。

- `group`：指定**分组的数据变量**。用于在绘制多个图形时将数据分组，并为每个组分配不同的图形属性。

- `label`：指定标签的数据变量。用于在图形中显示文本标签，如散点图中的数据标签。

```R
library(ggplot2)

# 示例数据集
df <- data.frame(x = 1:5, y = c(2, 4, 6, 8, 10))

# 创建绘图对象，定义映射关系
p <- ggplot(data = df, mapping = aes(x = x, y = y))
```

在上述示例中，我们首先创建了一个数据集 `df`，包含两列（x 和 y）。然后，使用 `ggplot()` 函数创建了一个基本的绘图对象 `p`，并在 `mapping` 参数中使用 `aes()` 函数定义了 x 和 y 列与图形的映射关系。


在进一步的绘图过程中，可以使用其他函数（如 `geom_*()`）来添加具体的几何对象，并利用之前定义的映射关系自动映射数据的变量到对应的图形属性。例如：

```R
# 添加点图层
p + geom_point()
```

这将在绘图对象 `p` 的基础上添加一个点图层，其中 x 和 y 列的数据将自动映射到点的位置。
通过 `aes()` 函数的灵活使用，可以根据需要将更多的数据变量与图形的视觉属性进行映射，以实现更多样化、富有信息的数据可视化。

### 关于ggplot()中说明aes(x = A, y = B)和在geom_point()说明aes(x = A, y = B)有什么区别?
全局映射关系和局部映射关系。没有局部映射以全局映射为准，有局部映射，参考局部映射，缺少得内容参考全局映射！

```R
library(ggplot2)
# 创建一个数据框
df <- data.frame(A = c(1, 2, 3, 4, 5),
                 B = c(2, 4, 1, 8, 10))
# 创建散点图
ggplot(data = df, aes(x = A, y = B)) +
  geom_smooth() +
  geom_line(aes(x = -A, y = B ,color="pink")) +
  geom_point(aes(x = A, y = B ,color="blue")) +
  geom_line() +
  geom_smooth(aes(x = -A))
```
![Pasted image 20231005115603](Pasted%20image%2020231005115603.png)

线图层与散点图层的映射关系不受全局映射关系的影响，我们在`geom_line()`函数中重新定义了局部映射关系。
这样，通过`ggplot()`函数和`geom_line()`函数的不同`aes()`定义，我们可以在同一个图形中同时显示散点图和线图，并为它们指定不同的映射关系和颜色。

### 为什么在aes()函数中定义的color显示得有些不对？
在您的代码中，`color`参数在`aes()`函数中被设置为**字符常量**，即`color = "black"`和`color = "pink"`。这样做会导致`color`被视为固定的常量，而不是映射到具体的数据变量。

要正确使用`aes()`函数中的`color`参数，应将它与实**际的数据变量**相关联，以便根据数据的不同值来确定颜色。在您的示例中，我将修正代码以正确映射颜色值：

```R
library(ggplot2)

# 创建一个数据框
df <- data.frame(A = c(1, 2, 3, 4, 5),
                 B = c(2, 4, 6, 8, 10),
                 C = c(22, 24, 26, 28, 30))

# 创建散点图和线图
ggplot(data = df) +
  geom_point(aes(x = A, y = B), color = "black") +
  geom_line(aes(x = A, y = C), color = "pink")
```

在修正后的代码中，我们将`color`参数直接放在`geom_point()`和`geom_line()`函数中，并将其设置为字符常量，如`color = "black"`和`color = "pink"`。这样，`color`参数被视为静态的颜色值，所有的散点和线都会被渲染为相应的颜色。

如果您希望根据**数据变量的不同值**来确定颜色，可以将`color`参数与实际的数据变量相关联，如`color = variable`。在这种情况下，您可以使用`scale_color_manual()`函数来手动指定颜色映射关系。以下是一个示例：
```R
ggplot(data = df) +
  geom_point(aes(x = A, y = B, color = "Variable 1")) +
  geom_line(aes(x = A, y = C, color = "Variable 2")) +
  scale_color_manual(values = c("Variable 1" = "black", "Variable 2" = "pink"))
```

在上述示例中，`color`参数与字符串变量相关联，如`color = "Variable 1"`和`color = "Variable 2"`。然后，使用`scale_color_manual()`函数手动指定了这些变量的颜色映射关系，分别为黑色和粉色。这样，散点和线将根据数据变量的不同值显示相应的颜色。