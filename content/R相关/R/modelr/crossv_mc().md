在R语言中，`modelr`包中的`crossv_mc()`函数用于执行基于多次交叉验证的模型评估。它将数据集划分为K个折叠（folds），并在每个折叠上进行模型训练和评估。
**函数定义**：
```R
crossv_mc(data, formula, model_fn, folds = 10, repeats = 1, stratify = FALSE, seed = NULL, ...)
```

**参数**：
- `data`：要用于交叉验证的数据集。
- `formula`：一个公式，用于指定模型的拟合公式。
- `model_fn`：一个函数，用于指定要拟合的模型。
- `folds`：交叉验证中的折叠（folds）数量。默认为10。
- `repeats`：重复进行交叉验证的次数。默认为1。
- `stratify`：一个逻辑值，用于指定是否按照目标变量的分布进行分层抽样。默认为FALSE。
- `seed`：一个整数，用于设置随机数种子，以确保可重复性。
- `...`：其他参数，用于传递给模型拟合函数。

**示例**：
```R
library(modelr)

# 示例：执行基于多次交叉验证的模型评估
data <- iris  # 使用iris数据集

# 定义模型拟合函数
model_fn <- function(data, formula) {
  lm(formula, data = data)
}

# 执行交叉验证
result <- crossv_mc(data, Sepal.Length ~ Sepal.Width + Species, model_fn, folds = 5, repeats = 3, stratify = TRUE)

# 打印交叉验证结果
print(result)
```

在示例中，我们使用`modelr`包中的`crossv_mc()`函数对`iris`数据集执行基于多次交叉验证的模型评估。我们指定了模型的拟合公式`Sepal.Length ~ Sepal.Width + Species`以及模型拟合函数`lm()`。我们将数据集分为5个折叠，并重复进行3次交叉验证。我们还将`stratify`参数设置为`TRUE`，以按照目标变量的分布进行分层抽样。最后，我们打印交叉验证的结果。

请注意，`modelr`包中的`crossv_mc()`函数允许您自定义模型拟合函数，并根据需要调整其他参数，以适应特定的模型评估需求。