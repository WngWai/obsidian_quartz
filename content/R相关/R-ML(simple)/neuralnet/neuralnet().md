```python
eneff_model <- neuralnet(f,data=cbind(eneff_train_pp,eneff_train_outputs_pp),hidden=10,act.fct="logistic",linear.output=TRUE,err.fct="sse",rep=1)
```

这个函数是 `neuralnet` 包的核心，用于创建和训练神经网络，得到一个**神经网络模型对象**

  ```r
  neuralnet(formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05, rep = 1, startweights = NULL, learningrate.limit = NULL, learningrate.factor = NULL, learningrate = NULL, lifesign = "none", lifesign.step = 1000, algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE)
  ```
  - **stepmax**：最大迭代次数。
  - **rep**：重复训练次数。
  - **startweights**：初始权重。
  - **learningrate.limit**、**learningrate.factor**、**learningrate**：学习率设置。
  - **lifesign**、**lifesign.step**：控制训练期间的输出。
  - **algorithm**：**训练算法**（默认是 "rprop+"）。

```r
neuralnet(formula, data, hidden = 1, threshold = 0.01, stepmax = 1e+05, rep = 1, startweights = NULL, learningrate.limit = NULL, learningrate.factor = NULL, 
          learningrate = NULL, algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", linear.output = TRUE, lifesign = "none", lifesign.step = 1000, 
          compute = TRUE, differentiable = TRUE, exclude = NULL, constant.weights = NULL, likelihood = FALSE)
```

常用参数详细介绍
- **formula**: 描述**模型的公式**。典型形式为 `y ~ x1 + x2 + x3`。指定**目标变量（因变量）和特征变量（自变量）**。

 - **data**：用于训练模型的数据集。其中包含**目标变量和特征变量**。

- **hidden**: **隐藏层节点数**的向量。默认值`1`，指定每个隐藏层的神经元数量。例如，`hidden = c(5, 3)` 表示第一隐藏层有 5 个神经元，第二隐藏层有 3 个神经元。

- **threshold**: 停止训练的阈值。 默认值`0.01`，当误差低于该阈值时停止训练。

- **stepmax**: 最大迭代次数。
  - **默认值**: `1e+05`
  - **说明**: 最大训练步数。

- **rep**: 网络训练的**重复次数**。默认 `1`。指定训练网络的重复次数，每次训练可能使用不同的随机初始权重。

- **startweights**: 初始权重的向量。
  - **默认值**: `NULL`
  - **说明**: 指定初始权重。如果为 `NULL`，则随机初始化。

- **learningrate.limit**: 学习速率的上下限。
  - **默认值**: `NULL`
  - **说明**: 指定学习速率的上下限，适用于学习速率变化算法。

- **learningrate.factor**: 学习速率因子。
  - **默认值**: `NULL`
  - **说明**: 指定学习速率调整因子，适用于学习速率变化算法。

- **learningrate**: 学习速率。
  - **默认值**: `NULL`
  - **说明**: 指定固定学习速率。

- **algorithm**: 使用的训练算法。
  - **默认值**: `"rprop+"`
  - **说明**: 可选值包括 `"backprop"`（反向传播）、`"rprop+"`（改进弹性传播）、`"rprop-"`（经典弹性传播）。

- **err.fct**: 误差函数。默认值`"sse"`，可选值包括 `"sse"`（均方误差）和 `"ce"`（交叉熵）。

- **act.fct**: 激活函数。默认值`"logistic"`可选值包括 `"logistic"` 和 `"tanh"`，表示使用 logistic 激活函数和 tanh 激活函数。

- **linear.output**: **输出层**是否**使用线性激活函数**。默认值 `TRUE`。当 `TRUE` 时，输出层使用线性激活函数，适用于回归任务。当 `FALSE` 时，输出层使用 `act.fct` 指定的激活函数，适用于分类任务。

- **lifesign**: 输出训练过程的进度信息。
  - **默认值**: `"none"`
  - **说明**: 可选值包括 `"none"`（无输出）、`"minimal"`（简单输出）、`"full"`（详细输出）。

- **lifesign.step**: 输出进度信息的步长。
  - **默认值**: `1000`
  - **说明**: 每隔 `lifesign.step` 步输出一次进度信息。

- **compute**: 是否计算训练后的网络输出。
  - **默认值**: `TRUE`
  - **说明**: 如果为 `TRUE`，训练后计算并返回网络输出。

- **differentiable**: 激活函数是否可微。
  - **默认值**: `TRUE`
  - **说明**: 指定激活函数是否需要可微分。

- **exclude**: 排除特定的连接权重。
  - **默认值**: `NULL`
  - **说明**: 可以通过一个矩阵来指定要排除的连接权重。

- **constant.weights**: 设置常量权重。
  - **默认值**: `NULL`
  - **说明**: 可以通过一个矩阵来指定常量权重。

- **likelihood**: 是否计算似然函数。
  - **默认值**: `FALSE`
  - **说明**: 如果为 `TRUE`，则计算并返回似然函数。

```r
library(neuralnet)

# 构造数据
data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  y = c(0, 1, 1, 0)
)

# 训练神经网络
nn <- neuralnet(
  y ~ x1 + x2,
  data = data,
  hidden = c(5, 3),
  threshold = 0.01,
  stepmax = 1e+05,
  rep = 3,
  algorithm = "rprop+",
  err.fct = "sse",
  act.fct = "logistic",
  linear.output = FALSE,
  lifesign = "minimal",
  lifesign.step = 1000
)

# 打印模型
print(nn)
```

上述代码展示了如何使用 `neuralnet` 包训练一个两层神经网络，并设置各个常用参数。

## 预测结果的属性介绍
print(nn) 模型打印的结果!带有不少属性！
```python
$call
neuralnet(formula = y ~ x1 + x2, data = data, hidden = c(3), 
    linear.output = FALSE)

$response
  y
1 0
2 1
3 1
4 0

$covariate
     x1 x2
[1,]  0  0
[2,]  0  1
[3,]  1  0
[4,]  1  1

$model.list
$model.list$response
[1] "y"

$model.list$variables
[1] "x1" "x2"


$err.fct
function (x, y) 
{
    1/2 * (y - x)^2
}
<bytecode: 0x000001c10fa24f70>
<environment: 0x000001c10fa22a20>
attr(,"type")
[1] "sse"

$act.fct
function (x) 
{
    1/(1 + exp(-x))
}
<bytecode: 0x000001c10fa47478>
<environment: 0x000001c10fa47e50>
attr(,"type")
[1] "logistic"

$linear.output
[1] FALSE

$data
  x1 x2 y
1  0  0 0
2  0  1 1
3  1  0 1
4  1  1 0

$exclude
NULL

$net.result
$net.result[[1]]
          [,1]
[1,] 0.4672222
[2,] 0.5530290
[3,] 0.4361072
[4,] 0.5189882


$weights
$weights[[1]]
$weights[[1]][[1]]
           [,1]       [,2]       [,3]
[1,] -0.7604756 -0.1294916  0.2609162
[2,] -0.4301775  0.3292877 -1.0650612
[3,]  1.3587083  1.5150650 -0.8868529

$weights[[1]][[2]]
           [,1]
[1,] -0.6456620
[2,]  1.0240818
[3,]  0.1598138
[4,]  0.2007715



$generalized.weights
$generalized.weights[[1]]
           [,1]      [,2]
[1,] -0.1350865 0.3185565
[2,] -0.1409623 0.3168346
[3,] -0.1114055 0.2706736
[4,] -0.1306595 0.3533120


$startweights
$startweights[[1]]
$startweights[[1]][[1]]
           [,1]       [,2]       [,3]
[1,] -0.5604756 0.07050839  0.4609162
[2,] -0.2301775 0.12928774 -1.2650612
[3,]  1.5587083 1.71506499 -0.6868529

$startweights[[1]][[2]]
           [,1]
[1,] -0.4456620
[2,]  1.2240818
[3,]  0.3598138
[4,]  0.4007715



$result.matrix
                              [,1]
error                  0.502701752
reached.threshold      0.007547241
steps                  3.000000000
Intercept.to.1layhid1 -0.760475647
x1.to.1layhid1        -0.430177489
x2.to.1layhid1         1.358708314
Intercept.to.1layhid2 -0.129491609
x1.to.1layhid2         0.329287735
x2.to.1layhid2         1.515064987
Intercept.to.1layhid3  0.260916206
x1.to.1layhid3        -1.065061235
x2.to.1layhid3        -0.886852852
Intercept.to.y        -0.645661970
1layhid1.to.y          1.024081797
1layhid2.to.y          0.159813827
1layhid3.to.y          0.200771451

attr(,"class")
[1] "nn"
          [,1]
[1,] 0.4361072
[2,] 0.5530290
```