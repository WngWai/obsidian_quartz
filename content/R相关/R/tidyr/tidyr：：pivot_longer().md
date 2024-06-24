在R语言中，`pivot_longer()`函数是tidyverse包中的函数，用于将**宽格式（wide format）的数据转换为长格式（long format）**。

```R
wide <- tribble(
  ~id,  ~x, ~y, ~z,
   1,  'a',  'c', 'e',
   2,  'b',  'd', 'f'
)
wide
```

![[Pasted image 20240320084751.png]]

```R
long <- wide %>%
  pivot_longer(
    cols = x:z,
    names_to = "variable",
    values_to = "value"
  )
long
```

![[Pasted image 20240320084758.png]]

**函数定义**：
```R
pivot_longer(data, cols, names_to, values_to, names_prefix = NULL, names_sep = NULL)
```

**参数**：
- `data`：要进行转换的数据框（data frame）或数据集。
- `cols`：**要转换的列名或列索引**，可以使用`-`表示排除某些列。

c(-column1, -column2)，表示排序data中这两列，对其他列进行操作！

- `names_to`：生成的**新列的名称**，通常是一个字符向量。
- `values_to`：生成的**新列中的值所在的列名**。
- `names_prefix`：可选参数，用于指定**列名的前缀**。
- `names_sep`：可选参数，用于指定**列名中的分隔符**。


**示例**：
以下是使用`pivot_longer()`函数将宽格式数据转换为长格式数据的示例：

```R
library(tidyverse)

# 示例数据框
data <- data.frame(
  id = c(1, 2, 3),
  A = c(10, 20, 30),
  B = c(15, 25, 35),
  C = c(18, 28, 38)
)

# 转换为长格式数据
long_data <- pivot_longer(data, cols = -id, names_to = "Category", values_to = "Value")

# 打印转换后的数据
print(long_data)
```

在上述示例中，我们首先加载了`tidyverse`包，其中包含了`pivot_longer()`函数。

然后，我们创建了一个示例数据框`data`，其中包含了"id"列和"A"、"B"、"C"三列作为数据。这是一个宽格式的数据。

接着，我们使用`pivot_longer()`函数将宽格式数据`data`转换为长格式数据。我们指定要转换的列为除了"id"列以外的其他列，将生成的新列命名为"Category"和"Value"。

最后，我们打印出转换后的数据`long_data`，它表示了宽格式数据转换为长格式数据后的结果。

以下是打印出的内容：

```
# A tibble: 9 x 3
     id Category Value
  <dbl> <chr>    <dbl>
1     1 A           10
2     1 B           15
3     1 C           18
4     2 A           20
5     2 B           25
6     2 C           28
7     3 A           30
8     3 B           35
9     3 C           38
```

在上述输出中，我们可以看到宽格式数据已经被转换为了长格式数据，"A"、"B"、"C"三列的值分别对应于新生成的"Category"列，而数值则对应于"Value"列。