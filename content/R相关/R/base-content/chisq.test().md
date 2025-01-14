在 R 语言中，`chisq.test()` 函数用于执行卡方检验（Chi-squared test）。卡方检验是一种用于评估观察值与期望值之间的差异的统计方法，常用于分析分类变量之间的关联性。下面是对 `chisq.test()` 函数的参数进行详细介绍和举例：

![Pasted image 20231120135431](Pasted%20image%2020231120135431.png)

**函数语法：**
```R
chisq.test(x, y = NULL, correct = TRUE, p = rep(1/length(x), length(x)), ...)
```

**参数说明：**

- `x`：一个数据向量或者一个数据矩阵。如果是向量，则表示**单个分类变量的观察值**；如果是矩阵，则表示**多个分类变量之间的关联表**。

c=(38, 24, 18, 18, 2)；期望值系统自动汇总观测值，乘以期望值的概率分布间接求得

- `y`：可选参数，当 `x` 是一个数据矩阵时，`y` 是一个数据向量，用于表示分类变量的**配对信息**。

- `correct`：一个逻辑值，用于指定是否应用连续性校正（continuity correction）。默认为 `TRUE`，表示应用连续性校正。

- `p`：一个概率向量，用于指定理论**期望值的概率分布**，即期望频次的概率。默认情况下，**每个分类的概率相等**

c=(0.40, 0.20, 0.20, 0.15, 0.05)


- `...`：其他参数，用于传递给 `chisq.test()` 函数的选项。

**返回值：**
函数返回一个包含卡方检验结果的对象，其中包括卡方统计量、自由度、p 值等。

**示例：**
下面是一个使用 `chisq.test()` 函数进行卡方检验的示例：

```R
# 创建一个分类变量的观察值向量
x <- c(10, 15, 5, 8)

# 使用 chisq.test() 进行卡方检验
result <- chisq.test(x)

# 打印卡方检验结果
print(result)
```

在上述示例中，我们创建了一个分类变量的观察值向量 `x`，表示不同类别的频数。然后，我们使用 `chisq.test()` 函数对观察值进行卡方检验。最后，我们打印出卡方检验的结果。

请注意，`chisq.test()` 函数还可以用于分析多个分类变量之间的关联性，此时 `x` 参数应该是一个数据矩阵，每一列代表一个分类变量的观察值。你可以根据实际需要传递不同的参数来执行相应的卡方检验，并根据检验结果进行数据分析和解释。

### 卡方的拟合优度检验
一个类别变量，检验样本能多大程度代表总体。
![Pasted image 20231224105442](Pasted%20image%2020231224105442.png)




### 卡方的独立性检验
两个类别变量，看类别间是否相互独立。H_1：两个类别变量不是相互独立，对备用假设有主观偏好，不然不会设置更为严格的证明条件，从而增加H1的说服力。

先构建类别频次矩阵
![Pasted image 20231120142431](Pasted%20image%2020231120142431.png)


![Pasted image 20231120142440](Pasted%20image%2020231120142440.png)