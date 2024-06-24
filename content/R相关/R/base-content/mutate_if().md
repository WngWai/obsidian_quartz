`mutate_if` 是 `dplyr` 包中的一个函数，用于在**满足特定条件的列上应用相关函数**。

```R
mutate_if(.data, .predicate, .funs, ...)
```

1. **.data** 是数据框或 tibble。
2. **.predicate** 是一个用来**测试哪些列应该被变换**的函数，通常是基于数据类型或其他条件。
3. **.funs** 是要应用的函数，可以是**简单函数或复杂表达式**。


如果该列为数值类型，则进行标准化操作！
```R
# 确保 USJ_dt 是 data.frame 或 tibble 类型
noteUmap1 <- USJ_dt %>%
  mutate_if(is.numeric, scale) %>%  # 如果 USJ_dt 中的数值类型列需要标准化，则开启 scale 参数
  mutate(UMAP1 = umap_result$layout[,1], UMAP2 = umap_result$layout[,2]) %>%  # 添加 UMAP 结果
  pivot_longer(names_to = "Feature", values_to = "Value", cols = -c(UMAP1, UMAP2, Status))  # 将宽格式数据转为长格式数据
```

在这个修改后的代码中，`mutate_if` 正确地应用于 `data.frame` 或 `tibble`，并且确保使用的是适合的参数和数据类型。如果 `USJ_dt` 需要从其他类型（例如列表或矩阵）转换为 `data.frame`，你可以使用 `as.data.frame(USJ_dt)` 或 `tibble::as_tibble(USJ_dt)` 来转换。这样保证了 `dplyr` 的函数可以正确执行。