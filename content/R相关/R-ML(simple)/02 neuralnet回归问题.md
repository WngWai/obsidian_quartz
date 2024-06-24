`neuralnet` 包是 R 语言中用于训练和使用人工神经网络的一个重要包。它提供了**简便**的方法来创建、训练和评估神经网络

```r
# 加载必要的包
library(neuralnet)

# 创建一个示例数据集
data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)  # XOR 逻辑
)

# 创建并训练神经网络
nn <- neuralnet(y ~ x1 + x2, data, hidden = c(3), linear.output = FALSE)

# 打印模型结果
print(nn)

# 对新数据进行预测
new_data <- data.frame(x1 = c(1, 0), x2 = c(0, 1))
predictions <- compute(nn, new_data)
print(predictions$net.result)

# 可视化神经网络
plot(nn)
```


模型创建与训练
[[neuralnet()]]
```python
# 快速构建一个公式
f <- as.formula(paste("heatLoad + coolLoad ~", 
                      paste(n[!n %in% c("heatLoad","coolLoad")], 
                            collapse = " + ")))
```

结果与预测
[[compute()]]

可视化：
[[plot()]]


模型评估：
- **result.matrix**：神经网络模型对象中的一个属性，用于存储训练的结果矩阵，包括权重、误差等信息。




