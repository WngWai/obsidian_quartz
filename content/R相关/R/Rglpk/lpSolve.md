```R
# 安装和加载lpSolve库
install.packages("lpSolve")
library(lpSolve)

# 定义目标函数的系数向量
obj.coef <- c(300, 500)

# 定义左侧矩阵（约束条件的系数矩阵）
lhs <- matrix(c(1, 2, 0, 2, 1, 2), nrow = 3, byrow = TRUE)

# 定义约束条件的右侧值
rhs <- c(4, 12, 18)
前提是加装lpSolve这个包
# 定义约束条件的符号（小于等于）
direction <- c("<=", "<=", "<=")

# 求解线性规划问题
lp_solution <- lp("max", obj.coef, lhs, direction, rhs)

# 输出结果
print(lp_solution)

# 最优解
optimal_solution <- lp_solution$solution
cat("Optimal solution: x1 =", optimal_solution[1], ", x2 =", optimal_solution[2], "\n")

# 最大利润
max_profit <- lp_solution$objval
cat("Maximum profit: $", max_profit, "\n")
```