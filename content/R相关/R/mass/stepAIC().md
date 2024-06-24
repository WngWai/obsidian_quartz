在R语言中，`stepAIC()`函数是`MASS`包中的函数，用于在线性回归模型中执行**逐步逐级模型选择**。

**函数定义**：
```R
stepAIC(object, scope, direction = c("both", "forward", "backward"),
        trace = 1, keep = NULL, steps = 1000)
```

**参数**：
- `object`：一个包含所有可能的预测变量的完整线性回归模型对象。
- `scope`：一个包含预测变量的嵌套模型的公式，用于指定逐步逐级模型选择的搜索空间。
- `direction`：可选参数，指定逐步逐级模型选择的方向。可选值为"both"（默认，前向和后向选择）、"forward"（仅前向选择）和"backward"（仅后向选择）。
- `trace`：可选参数，控制详细程度的整数值，用于输出每个步骤的信息。默认值为1，输出详细信息。
- `keep`：可选参数，用于指定要保留的变量的名称。
- `steps`：可选参数，控制逐步逐级模型选择的步骤数，默认为1000。

**示例**：
以下是使用`stepAIC()`函数执行逐步逐级模型选择的示例：

```R
library(MASS)

# 示例数据集
data <- iris

# 构建初始线性回归模型
lm_model <- lm(Sepal.Length ~ ., data = data)

# 执行逐步逐级模型选择
stepwise_model <- stepAIC(lm_model, direction = "both")

# 打印选择后的模型
print(stepwise_model)
```

在上述示例中，我们首先加载了`MASS`包，其中包含了`stepAIC()`函数。

然后，我们使用`iris`数据集创建了一个示例数据集`data`，其中包含了多个预测变量和一个因变量。

接着，我们使用`lm()`函数构建了一个初始的线性回归模型`lm_model`，其中因变量是`Sepal.Length`，自变量是所有其他变量。

最后，我们使用`stepAIC()`函数对初始模型`lm_model`执行逐步逐级模型选择，默认进行前向和后向选择。选择后的模型保存在`stepwise_model`中，并打印出来。

以下是打印出的内容，显示选择后的模型：

```
Call:
lm(formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, 
    data = data)

Coefficients:
 (Intercept)  Sepal.Width  Petal.Length   Petal.Width  
      1.8551       0.6508       0.7091      -0.5565  
```

在上述输出中，我们可以看到经过逐步逐级模型选择后的模型，其中包含了选择的自变量和它们的系数。