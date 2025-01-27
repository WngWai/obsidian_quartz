在R语言中，`qqnorm()`函数用于**绘制一个样本的正态概率图（QQ图，Quantile-Quantile Plot）**。

用于**评估数据是否符合正态分布**的图形方法。它对比了**观测值的分位数与正态分布的分位数**之间的关系，可以帮助我们判断数据是否近似服从正态分布。

![Pasted image 20231116141335](Pasted%20image%2020231116141335.png)

**函数定义**：
```R
qqnorm(y, ...)
```

**参数**：
- `y`：一个向量，表示要绘制QQ图的样本数据。
- `...`：其他可选参数，用于控制图形的外观，如颜色、标记符号等。

**示例**：
以下是使用`qqnorm()`函数绘制样本数据的QQ图的示例：

```R
# 创建示例样本数据
data <- rnorm(100)

# 绘制QQ图
qqnorm(data)
```

在上述示例中，我们首先使用`rnorm()`函数生成了一个包含100个服从标准正态分布的随机数的示例样本数据。

然后，我们使用`qqnorm()`函数对样本数据进行绘制，不指定其他参数。

这将绘制一个QQ图，显示样本数据的分位数和标准正态分布的分位数之间的比较关系。

根据样本数据和标准正态分布之间的相似性，我们可以判断样本数据是否近似服从正态分布。

输出结果是一个QQ图，其中点位于一条直线上，表示样本数据近似服从正态分布。