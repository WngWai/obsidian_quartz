核心是对学习器中的参数设置范围，本质是对学习算法设置参数调整范围？学习器不同，参数也会有差异！

`paradox` 是 R 语言中用于**处理参数和参数空间**的包，它是 `mlr3` 生态系统的一部分，专注于提供高级的参数描述和操作工具。

超参数搜索空间（Search Space）你可以使用 `ps()` 从 `paradox` 包中创建参数集合以定义超参数的范围和类型。

## 参数空间（重点）
### 参数搜索空间

- ps()创建一个搜索空间对象。Domain类对象？
   
	`p_int()`: 创建一个整数型参数。
	`p_dbl()`: 创建一个浮点型参数。
	`p_fct()`: 创建一个因子型参数。
	`p_lgl()`: 创建一个逻辑型参数。
	`p_ctrl()`: 创建一个控制型参数。

### 待整理
1，**算法的选择和配置**
   - `set_algo()`: 配置算法以在搜索空间中优化超参数。
   - `add_algo()`: 向搜索空间中添加一个待优化的算法。

   ```R
   set_algo(search, algo = "random_search", max_evals = 100)
   ```
   这个示例将搜索空间中的优化算法设置为随机搜索，并设置最大评估次数为100次。

2，**采样方法的选择和配置**：
   - `set_sampler()`: **配置采样方法**以在搜索空间中进行采样。
   - `add_sampler()`: 向搜索空间中**添加一个待采样的方法**。
   ```R
   set_sampler(search, sampler = "grid_search", resolution = 10)
   ```
   这个示例将搜索空间中的采样方法设置为网格搜索，并设置分辨率为10。

4. **算法和采样方法的优化**：
   - `optimize()`：在给定的搜索空间中执行超参数优化。
   - `sample()`：在给定的搜索空间中执行参数采样。
   ```R
   result = optimize(search, objective_function)
   ```
   
   这个示例在给定的搜索空间中执行超参数优化，并将优化结果存储在`result`变量中。

**结果的获取和可视化**：
   - `get_result()`: 获取优化或采样的**结果**。
   - `plot_result()`: 绘制优化或采样的**结果图**。

```R
best_params = get_result(result, "best_params")
plot_result(result)
```
这个示例从优化结果中获取最佳的超参数组合。
这个示例将优化结果绘制成图形，展示超参数的性能和收敛情况。


## 参数集合

- ParamSet$new()创建一个**新的参数集合**。Param类对象？

	`ParamDbl$new()`: 定义一个**双精度型**参数。
	`ParamInt$new()`: 定义一个**整型**参数。
	`ParamLgl$new()`: 定义一个**逻辑型**参数（真/假）。
	`ParamFct$new()`: 定义一个**因子型**参数，用于有限选择的情况。
	`ParamUty$new()`: 定义一个**实用型**（"untyped"）参数，可以是任何类型。

### 2. 参数约束和依赖
[[add_dep()]]给参数**添加依赖**关系，指定一个参数的值或有效性依赖于另一参数的值。
[[add_constraint()]]给参数集合**添加约束**。用于定义参数之间更复杂的关系，比如两个参数的和不超过一个特定值。

### 3. 参数空间操作
- `generate_design()`: 在指定的参数空间内生成一个设计，通常用于参数优化过程中的采样。
- `generate_design_random()`: 在参数空间内随机生成一个设计。
- `filter_design()`: 根据给定的条件过滤一个设计。

### 4. 参数值的验证和转换
- `test_param_set()`: 测试给定的参数值是否满足参数集合的定义，包括约束和依赖。
- `trafo()`: 应用转换函数到参数或参数集合。常用于将参数从原始空间映射到实际使用的空间，例如对数空间到线性空间。

### 5. 辅助函数
- `has_constraints()`: 检查参数集合是否有约束。
- `has_deps()`: 检查参数集合是否有依赖关系。
- `length()`: 返回参数集合中参数的数量。
- `get_values()`, `set_values()`: 获取或设置参数集合中参数的值。

### 使用示例：
以下是一个创建参数集合并添加参数的简单示例：
```r
library(paradox)

# 创建参数集合
ps <- ParamSet$new(list(
  ParamDbl$new("learning_rate", lower = 0.01, upper = 0.3),
  ParamInt$new("num_trees", lower = 100, upper = 1000),
  ParamFct$new("criterion", levels = c("gini", "entropy"))
))

# 添加依赖关系
ps$add_dep("num_trees", on = "criterion", CondEqual$new("gini"))

# 添加约束
ps$add_constraint("learning_rate + num_trees / 1000 <= 1")

# 打印参数集合的概述
print(ps)
```

`paradox` 包提供的这些工具和功能使其成为描述和操作复杂参数空间的强大资源，特别适用于超参数优化和模型配置的场景。


## 参数空间和参数集合的差异
两种说法：

1，两者本质相同，只是形式差异！
```R
param_set <- ParamSet$new(list(
  ParamDbl$new("learning_rate", lower = 0.01, upper = 0.3),
  ParamInt$new("num_trees", lower = 100, upper = 1000),
  ParamFct$new("criterion", levels = c("gini", "entropy"))
))

param_set <- ps(
  learning_rate = p_dbl(0.01, 0.3),
  num_trees = p_int(100, 1000),
  criterion = p_fct(levels = c("gini", "entropy"))
)
```

2，前者参数可变，后者参数固定（确定了范围）；
```R
degree <- 2  # 多项式的阶数，可根据需要进行更改
param_set <- ParamSet$new(
  lapply(1:degree, function(i) ParamDbl$new(paste0("coef_", i), lower = -1, upper = 1))
)


param_set <- ps(
  centers = p_int(2, 9),
  algorithm = p_fct(levels = c("Hartigan-Wong", "Lloyd", "MacQueen")),
  nstart = p_int(10, 10)
)
```

