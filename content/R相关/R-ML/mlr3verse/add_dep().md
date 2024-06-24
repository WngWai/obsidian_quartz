在 `paradox` 包中，参数类的方法 `add_dep()` 用于为参数添加依赖关系。这意味着参数的有效性可以依赖于其他参数的值。使用 `add_dep()` 可以定义这样的依赖关系，并且可以指定一系列的规则来控制参数之间的交互。

### 方法定义

`add_dep()` 方法通常定义如下：

```r
add_dep(id, on, cond, action = paradox::requires)
```

其中各参数的含义如下：

- `id`: 参数的名称，它是要添加依赖性的参数的标识符。
- `on`: 依赖的参数名称，`id` 参数的有效性取决于这个 `on` 参数。
- `cond`: 一个条件表达式，定义了 `on` 参数应当满足的条件。
- `action`: 依赖动作，默认为 `paradox::requires`，意味着 `id` 的有效性要求 `on` 参数满足 `cond` 条件。

### 属性介绍

- **id**: 你想要添加依赖关系的参数的名称。
- **on**: 与 `id` 参数有依赖关系的参数名称。
- **cond**: 一个条件，它是一个 `paradox::Condition` 对象，指定了 `on` 参数必须满足 `id` 参数才有效的条件。
- **action**: 依赖关系的类型，默认是 `requires`，表示 `id` 参数的存在依赖于 `on` 参数满足 `cond` 条件。还可以是 `paradox::forbids`，表示当 `on` 参数满足 `cond` 条件时，`id` 参数是不允许的。

### 应用举例

假设我们有一个包含两个参数 `param1` 和 `param2` 的参数集合，并且我们想要 `param2` 只有在 `param1` 值为 `TRUE` 时才有效。我们可以使用 `add_dep()` 来实现这个依赖关系：

```r
library(paradox)

# 创建一个参数集合
ps <- ParamSet$new(list(
  ParamLgl$new("param1"),
  ParamDbl$new("param2", lower = 0, upper = 1)
))

# 添加依赖关系
ps$add_dep("param2", on = "param1", cond = CondEqual$new(TRUE))

# 打印参数集合，查看依赖关系
print(ps)
```

在这个例子中，`param2` 依赖于 `param1` 的值。如果 `param1` 是 `TRUE`，那么 `param2` 就是有效的；如果 `param1` 是 `FALSE`，那么设置 `param2` 的值会导致错误，因为它不满足依赖条件。

使用 `add_dep()` 可以帮助我们定义复杂的参数关系和约束，这对于确保参数的有效组合和构建具有条件逻辑的模型非常有用。