 是 R 语言中的一个内置函数，用于计算**正态分布的累积分布函数**（CDF）。也就是概率密度函数
`pnorm()` 函数的语法如下：
```R
pnorm(x, mean = 0, sd = 1, lower.tail = TRUE)
```

参数说明如下：
- `x`：要计算累积分布函数的数值。**

- `mean`：正态分布的均值（默认为 0）。

- `sd`：正态分布的标准差（默认为 1）。

- `lower.tail`：一个逻辑值，表示是否计算**累积分布函数的下尾概率**（默认为 `TRUE`）。如果为 `TRUE`，将计算 P(X ≤ x)；如果为 `FALSE`，将计算 P(X > x)，反着来的。

当`lower.tail = TRUE`时，表示计算累积分布函数的值，即计算从负无穷到给定分位数的概率。

当`lower.tail = FALSE`时，表示计算分位数，即计算给定概率下的分位数，对应的尾部概率为从给定分位数到正无穷的概率。
![Pasted image 20230924114211](Pasted%20image%2020230924114211.png)
下面是一个示例，展示如何使用 `pnorm()` 函数计算正态分布的累积分布函数：

```R
# 计算标准正态分布在 x = 1 处的累积分布函数值
cdf <- pnorm(1)

# 打印结果
print(cdf)
```

在上述示例中，我们使用 `pnorm()` 函数计算标准正态分布在 x = 1 处的累积分布函数值，并将结果存储在名为 `cdf` 的变量中。

然后，我们通过打印 `cdf` 来查看计算得到的累积分布函数值。

请注意，`pnorm()` 函数计算的是标准正态分布的累积分布函数。如果需要计算其他均值和标准差的正态分布的累积分布函数，可以通过调整 `mean` 和 `sd` 参数的值来实现。


### 离群点
μ加减3sigm，超出外的数据称为离群点。
![Pasted image 20230924112707](Pasted%20image%2020230924112707.png)