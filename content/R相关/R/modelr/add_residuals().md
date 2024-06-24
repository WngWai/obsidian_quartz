在R语言中，`modelr`包中的`add_residuals()`函数用于将**模型的残差（residuals）添加到数据框中作为新的一列**。残差是指模型预测值与观测值之间的差异，通过将残差添加到数据框中，可以方便地进行进一步的分析和可视化。
**函数定义**：
```R
add_residuals(data, model, ..., name = "residuals")
```

**参数**：
- `data`：包含**输入特征**的数据框或数据集。
- `model`：**拟合的模型对象**，例如lm()、glm()等。而非optim()，可能是optim的结果已经很直观了，可以直接知道目标函数的最优解是多少！
- `...`：其他参数，用于传递给模型的残差计算函数。
- `name`：**新列的名称**，默认为"residuals"。

**示例**：
```R
library(modelr)

# 示例：将模型的残差添加到数据框中
data <- data.frame(
  x = c(1, 2, 3, 4),
  y = c(2, 4, 6, 8)
)

# 拟合线性回归模型
model <- lm(y ~ x, data = data)

# 添加残差列
data <- add_residuals(data, model)

# 打印更新后的数据框
print(data)
```

在示例中，我们使用`modelr`包中的`add_residuals()`函数将线性回归模型的残差添加到数据框中。我们创建了一个包含输入特征`x`和观测值`y`的数据框，并将其传递给`data`参数。然后，我们使用`lm()`函数拟合了一个线性回归模型，并将其传递给`model`参数。我们调用`add_residuals()`函数来添加模型的残差列，默认列名为"residuals"。最后，我们打印更新后的数据框，其中包含了模型的残差列。

请注意，您可以根据需要传递其他参数给残差计算函数，例如使用`type = "response"`来返回回归模型的残差。此外，您可以通过指定`name`参数来自定义新列的名称。

### model参数的形式
![Pasted image 20231015141611](Pasted%20image%2020231015141611.png)