`nnet` 包是 R 语言中用于训练和使用**多层前馈神经网络（MLP）** 的一个重要包。

```r
# 加载必要的包
library(nnet)

# 创建一个示例数据集
data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)  # XOR 逻辑
)

# 创建并训练神经网络
nn <- nnet(y ~ x1 + x2, data = data, size = 3, maxit = 200, linout = FALSE)

# 打印模型结果
print(nn)

# 对新数据进行预测
new_data <- data.frame(x1 = c(1, 0), x2 = c(0, 1))
predictions <- predict(nn, new_data, type = "class")
print(predictions)

# 输出模型摘要
summary(nn)
```

模型创建与训练：
[[nnet()]]

[[multinom()]] 解决多分类问题，softmax回归



结果与预测：
**predict()**：用于对新数据进行预测。
  ```r
  predict(object, newdata, type = c("raw", "class"))
  ```

  - **object**：一个神经网络模型对象（由 `nnet` 函数生成）。
  - **newdata**：用于预测的新数据集。
  - **type**：预测类型，可以是 `raw`（默认）或 `class`。

模型评估：
**summary()**：用于生成模型的总结信息。
  ```r
  summary(object)
  ```
  - **object**：一个神经网络模型对象。



