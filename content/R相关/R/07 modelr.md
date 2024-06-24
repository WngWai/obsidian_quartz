`modelr`是R语言中的一个包，它提供了一些用于**建模和模型评估**的工具和函数。`modelr`包的主要功能包括以下几个方面：

[[add_residuals()]] 残差
[[lm()]]拟合线性回归模型
[[glm()]]拟合广义线性回归模型

1. 数据分割与重抽样（Data Splitting and Resampling）：
[[R相关/R/modelr/resample()]]: 对数据集进行重抽样，生成用于模型训练和评估的训练集和测试集。
[[bootstraps()]]: 使用自助法（bootstrap）生成多个数据集，用于模型的自助聚合和评估。

2. 模型评估（Model Evaluation）：
[[crossv_mc()]]: 执行**交叉验证**，计算给定模型在多次交叉验证中的性能指标。
[[rmse()]]: 计算**均方根误差**（Root Mean Squared Error）。
[[mae()]]: 计算**平均绝对误差**（Mean Absolute Error）。
[[rsq()]]: 计算**决定系数**（R-squared）。
[[mape()]]: 计算**平均绝对百分比误差**（Mean Absolute Percentage Error）

