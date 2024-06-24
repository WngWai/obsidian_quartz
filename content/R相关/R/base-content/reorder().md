在R语言中，`reorder()`函数用于**重新排序一个因子变量（或字符向量）**，以便**根据另一个变量的值进行排序**。它返回一个**重新排序后的因子变量**。

```r
reorder(x, X, FUN = mean, ..., order = is.ordered(x))
```

其中参数含义如下：

- `x`: 要重排序的因子（factor）。必须是因子类型吗？
- `X`: 与`x`长度相同的数值向量，`reorder`将根据这个向量进行排序。
也可以是待进行聚合操作的变量

- `FUN`: 一个函数，用于对每个因子水平下的`X`值进行某种计算，默认为`mean`。这个计算结果将用于确定新的因子水平顺序。
```R
ggplot(data =mpg)+
	geom_boxplot(mapping = aes(x = reorder(class,hwy,FUN= median),y =hwy))
```
class是离散型变量
hwy是连续型变量
fun应该是对class分类的连续型变量hwy进行统计计算

- `...`: 额外的参数可以传递给`FUN`函数。
- `order`: 一个逻辑值，表明是否保持因子的有序性。


```R
# 创建一个数据框
data <- data.frame(
  category = c("A", "B", "C", "D"),
  value = c(10, 20, 15, 25)
)

# 使用reorder()函数重新排序因子变量
data$category <- reorder(data$category, data$value)

print(data)
```

在上面的示例中，我们创建了一个包含分类变量`category`和数值变量`value`的数据框。
然后，我们使用`reorder()`函数重新排序了`category`因子变量，根据`value`变量的值进行排序。
在这个示例中，`reorder()`函数默认使用`mean`函数计算排序值，即根据每个`category`值对应的`value`的平均值进行排序。

打印输出的结果如下：
```
  category value
1        A    10
2        C    15
3        B    20
4        D    25
```
可以看到，`category`变量被重新排序，按照`value`变量的值从小到大进行排序。
`reorder()`函数对于创建基于其他变量排序的图表非常有用，例如基于某个指标值的堆叠条形图或线图。
您可以根据需要使用其他函数来计算排序值，并修改`...`参数以指定其他用于排序的变量。


### 也能实现df数据的重复排布
```R
property_name$property_region <- reorder(property_name$property_region , property_name$sum_property )
```


#### 示例1: 重排序条形图的条形

假设我们有一个数据框，记录了不同品牌汽车的油耗（mpg）和品牌（brand）。我们想要根据每个品牌汽车油耗的平均值来对条形图的条形进行排序。

```r
library(ggplot2)

# 创建示例数据
df <- data.frame(
  mpg = c(21, 22, 23, 20, 30, 31, 32, 29),
  brand = factor(c("BrandA", "BrandA", "BrandA", "BrandA", "BrandB", "BrandB", "BrandB", "BrandB"))
)

# 重排序品牌因子
df$brand <- reorder(df$brand, df$mpg, FUN = mean)

# 绘制条形图
ggplot(df, aes(x = brand, y = mpg)) +
  geom_bar(stat = "summary", fun = "mean") +
  theme_minimal() +
  labs(x = "Brand", y = "Average MPG", title = "Average MPG by Brand")
```
在这个例子中，`reorder()`函数根据品牌的平均油耗重新排序了`brand`因子的水平。然后，`ggplot2`的`geom_bar()`函数根据这个新的顺序绘制条形图，展示了按平均油耗排序的品牌。

#### 示例2: 使用中位数来重排序

如果我们想要根据中位数而不是均值来排序，只需要改变`FUN`参数：

```r
# 使用中位数重排序
df$brand <- reorder(df$brand, df$mpg, FUN = median)

# 绘制条形图
ggplot(df, aes(x = brand, y = mpg)) +
  geom_bar(stat = "summary", fun = "mean") +
  theme_minimal() +
  labs(x = "Brand", y = "Average MPG", title = "Average MPG by Brand Ordered by Median")
```

这里，`reorder()`函数使用`median`函数来计算每个品牌的中位数油耗，并根据这个值来重新排序因子水平。这种方式可能对于偏斜分布的数据更有意义。