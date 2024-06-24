`apply()` 函数是 R 语言中一个非常强大和常用的函数,它允许**对数组或矩阵的行或列应用某种函数**。

```R
apply(X, MARGIN, FUN, ...)
```

参数介绍:
- `X`: 需要进行操作的**数组或矩阵**
- `MARGIN`: 指定在哪个维度上应用函数,1表示行,2表示列,c(1,2)表示同时应用于行和列
- `FUN`: 要应用的函数,可以是**内置函数或自定义函数**

- `...`: 传递给 `FUN` 函数的其他参数


应用举例:
1. 计算矩阵每一列的平均值:
```r
matrix <- matrix(rnorm(12), nrow=4, ncol=3)
apply(matrix, 2, mean)
```

2. 计算矩阵每一行的标准差:
```r
apply(matrix, 1, sd)
```

3. 对数据框的每一列应用 `sum()` 函数:
```r
df <- data.frame(a=1:5, b=6:10, c=11:15)
apply(df, 2, sum)
```

4. 自定义函数应用到矩阵:
```r
custom_fun <- function(x) {
  return(x^2 + 1)
}
apply(matrix, c(1,2), custom_fun)
```
