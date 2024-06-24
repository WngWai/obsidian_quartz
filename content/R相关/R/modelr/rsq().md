在R语言中，`modelr`包中的`rsq()`函数用于计算模型的决定系数（R-squared）。决定系数是一种常用的模型评估指标，用于衡量模型对观测值的拟合程度。
**函数定义**：
```R
rsq(data, truth, estimate, na_rm = FALSE, ...)
```

**参数**：
- `data`：包含观测值和模型预测值的数据框或数据集。
- `truth`：观测值的列名或索引。
- `estimate`：模型预测值的列名或索引。
- `na_rm`：一个逻辑值，用于指定是否忽略包含缺失值的观测对。默认为FALSE。
- `...`：其他参数，用于传递给`na.rm`参数。

**示例**：
```R
library(modelr)

# 示例：计算模型的决定系数
data <- data.frame(
  truth = c(2, 4, 5, 7),
  estimate = c(2.5, 3.8, 4.2, 6.5)
)

# 计算决定系数
result <- rsq(data, truth, estimate)

# 打印决定系数
print(result)
```

在示例中，我们使用`modelr`包中的`rsq()`函数计算模型的决定系数。我们创建了一个包含观测值和模型预测值的数据框，并将其传递给`data`参数。我们指定了观测值的列名`truth`和模型预测值的列名`estimate`。然后，我们调用`rsq()`函数计算决定系数，并存储结果。最后，我们打印决定系数。

请注意，`modelr`包中的`rsq()`函数还允许您通过设置`na_rm`参数为`TRUE`来忽略包含缺失值的