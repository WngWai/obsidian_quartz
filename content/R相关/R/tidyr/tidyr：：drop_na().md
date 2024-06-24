在 R 语言中，`drop_na()` 函数通常属于 `tidyr` 包，该包提供了用于数据清理和整理的函数。`drop_na()` 主要用于删除数据框中包含缺失值（NA）的行。

**所属的包：** `tidyr`

**功能：** 删除数据框中包含缺失值（NA）的行。

```R
# 使用 tidyr 包
library(tidyr)

# 创建一个示例数据框
data <- data.frame(
  ID = c(1, 2, NA, 4),
  Name = c("Alice", "Bob", "Charlie", "David")
)

# 使用 drop_na() 删除包含缺失值的行
cleaned_data <- drop_na(data)

# 打印结果
print(cleaned_data)

# 输出：
  ID   Name
1  1  Alice
2  2    Bob
4  4  David
```

**定义：**
```R
drop_na(data, ...)
```

**参数介绍：**
- `data`：要处理的数据框。
- `...`：其他参数，用于**指定要考虑的列**。可以是**列名或列的位置**。

**返回值：**
返回一个新的数据框，其中删除了包含缺失值的行。

在这个例子中，`drop_na(data)` 使用 `tidyr` 包的函数，删除了数据框 `data` 中包含缺失值的行，得到了一个新的数据框 `cleaned_data`。在结果中，原始数据框中包含缺失值的行被删除了。