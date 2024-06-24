```python
dummies <- dummyVars(~ gender, data = df)

# 输出转换规则
Formula: ~.
10 variables, 0 factors
Variables and levels will be separated by '.'
A less than full rank encoding is used


# 转换规则可以在指定的类似书数据集上使用
dummy_df <- data.frame(predict(dummies, newdata = df))
```

`重复利用，但设计的真的繁琐！`

用于**创建虚拟变量（也称为哑变量或指示变量）转换规则**。这些虚拟变量通常用于将分类变量转换为数值变量，以便在回归模型或其他机器学习模型中使用。

```r
dummyVars(formula, data, fullRank = FALSE, levelsOnly = FALSE)
```

- `formula`: 一个使用 `~` 符号的公式,描述需要转换为哑变量的分类变量。例如 `~ variable1 + variable2`。

**target ~ .** 可以将数据集中除target变量外的数据全部转换为哑变量！

- `data`: 包含分类变量的数据框。
- `fullRank`: 逻辑值,决定是否创建完全秩的哑变量矩阵。默认为 `FALSE`。
- `levelsOnly`: 逻辑值,决定是否只返回分类变量的水平,而不返回哑变量矩阵。默认为 `FALSE`。

**应用示例**:

假设有一个数据框 `df` ,我们想将 `gender` 变量转换为哑变量:
```r
library(caret)

df <- data.frame(
  gender = c("male", "female", "male", "female"),
  age = c(25, 35, 30, 40),
  income = c(50000, 60000, 45000, 55000)
)

# 创建完全秩的哑变量矩阵
dummy_vars <- dummyVars(~ gender, data = df)
dummy_df <- data.frame(predict(dummy_vars, newdata = df))
str(dummy_df)
# 输出:
# 'data.frame':    4 obs. of  2 variables:
#  $ gender.female: num  0 1 0 1
#  $ gender.male   : num  1 0 1 0

# 只返回分类变量的水平
dummyVars(~ gender, data = df, levelsOnly = TRUE)
# 输出: 
# [1] "female" "male"
```

在上述示例中:
- 我们首先使用 `dummyVars()` 函数创建了完全秩的哑变量矩阵,并将其应用到原始数据 `df` 上得到新的数据框 `dummy_df`。
- 然后我们设置 `levelsOnly = TRUE` 仅返回了分类变量 `gender` 的水平。

使用 `dummyVars()` 函数可以方便地**将分类变量转换为机器学习算法所需的数值型特征**,是数据预处理的一个重要步骤。




## 几种写法的区分
我们来详细说明 `dummyVars` 函数的三种不同写法及其区别。

- **`dummyVars(heatLoad + coolLoad ~ ., data = eneff)`**：只将 `heatLoad` 和 `coolLoad` 作为因变量，其他所有变量作为自变量，并将自变量中的分类变量转换为虚拟变量。
- **`dummyVars(~heatLoad + coolLoad , data = eneff)`**：只转换 `heatLoad` 和 `coolLoad` 变量，不涉及其他变量。这种情况下，数据没有任何变化，因为 `heatLoad` 和 `coolLoad` 是数值型变量。
- **`dummyVars(~ ., data = eneff)`**：转换 `eneff` 数据框中的所有变量。分类变量会被转换为虚拟变量
### 数据准备

首先，假设我们有一个名为 `eneff` 的数据框，包含以下列：

- `heatLoad`: 数值型变量。
- `coolLoad`: 数值型变量。
- `buildingType`: 分类变量，取值为 "Residential", "Commercial"。
- `region`: 分类变量，取值为 "North", "South", "East", "West"。

```r
library(caret)

# 示例数据
eneff <- data.frame(
  heatLoad = c(100, 150, 200, 250),
  coolLoad = c(50, 70, 90, 110),
  buildingType = c("Residential", "Commercial", "Residential", "Commercial"),
  region = c("North", "South", "East", "West")
)
```

### 1. `dummyVars(heatLoad + coolLoad ~ ., data = eneff)`

这种写法表示只将 `heatLoad` 和 `coolLoad` 作为因变量，其他所有变量作为自变量，并将自变量中的分类变量转换为虚拟变量。

```r
dummy <- dummyVars(heatLoad + coolLoad ~ ., data = eneff)
transformed <- predict(dummy, newdata = eneff)
print(transformed)
```

在这个例子中，`buildingType` 和 `region` 是自变量，会被转换成虚拟变量：

```
  buildingType.Commercial buildingType.Residential region.East region.North region.South region.West
1                        0                        1           0            1            0           0
2                        1                        0           0            0            1           0
3                        0                        1           1            0            0           0
4                        1                        0           0            0            0           1
```

### 2. `dummyVars(~heatLoad + coolLoad , data = eneff)`

这种写法表示只转换 `heatLoad` 和 `coolLoad` 变量，不涉及其他变量。因为 `heatLoad` 和 `coolLoad` 是数值型变量，本身不会有任何变化。

```r
dummy <- dummyVars(~heatLoad + coolLoad, data = eneff)
transformed <- predict(dummy, newdata = eneff)
print(transformed)
```

输出结果：

```
  heatLoad coolLoad
1      100       50
2      150       70
3      200       90
4      250      110
```

这种情况下，数据没有任何变化，因为 `heatLoad` 和 `coolLoad` 是数值型变量。

### 3. `dummyVars(~ ., data = eneff)`

这种写法表示转换 `eneff` 数据框中的所有变量。分类变量会被转换为虚拟变量，但数值型变量保持不变。

```r
dummy <- dummyVars(~ ., data = eneff)
transformed <- predict(dummy, newdata = eneff)
print(transformed)
```

输出结果：

```
  heatLoad coolLoad buildingType.Commercial buildingType.Residential region.East region.North region.South region.West
1      100       50                        0                        1           0            1            0           0
2      150       70                        1                        0           0            0            1           0
3      200       90                        0                        1           1            0            0           0
4      250      110                        1                        0           0            0            0           1
```
