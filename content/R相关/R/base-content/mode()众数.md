在 R 语言中，没有内置函数直接计算众数（mode），因为众数不像均值或中位数那样是基本的统计量。众数是数据集中出现次数最多的值。但是，你可以用几种方法来计算众数。

### 单个众数的情况
一种简单的方法是使用 `table()` 函数来构建**频率表**（因子类型？？？），然后使用 `which.max()` 函数找到最大值**对应的索引**，像这样：
1，推荐
```R
names(which.max(table(data_q1$viewers))
```

```R
d <- c(4,5,3,1,6,7,8,9,11,12,32,1,3,4,5,5)

table(d) # 生成频率表，带有向量名和对应频率值
which.max(table(d)) # 上面是名称5，下面最大频率对应的索引值4
names(which.max(table(d))) # 取名称

names(table(d))
```

![Pasted image 20231226164828](Pasted%20image%2020231226164828.png)

```R
d <- c(2,1,3,4,5,6,7,8,9,0,11,12,32,1,3,4)

table(d) # 生成频率表，带有向量名和对应频率值
which.max(table(d)) # 上面是名称1，下面最大频率对应的索引值2
names(which.max(table(d))) 

names(table(d))
```

![Pasted image 20231226131407](Pasted%20image%2020231226131407.png)





2，
```r
# 定义一个计算众数的函数
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# 创建一个示例向量
x <- c(1, 2, 2, 3, 3, 3, 4, 4, 4, 4)

# 调用自定义的众数函数
mode_value <- Mode(x)
print(mode_value)
```

如果你有 `dplyr` 包安装，也可以使用 `dplyr` 的功能来计算众数，例如：

```r
library(dplyr)

# 使用 dplyr 计算众数
mode_value <- x %>%
  as.data.frame() %>%
  group_by(value = x) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  slice(1) %>%
  pull(value)

print(mode_value)
```

请注意，如果数据中有多个众数（也就是说，有多个值拥有相同的最高频率），上述两种方法都只会返回其中一个。如果你想要返回所有众数，需要对函数进行修改，以便它能够处理这种情况。

### 多个众数的情况
在存在多个众数的情况下，你可以通过编写一个自定义的 R 函数来找出所有的众数。这个函数将使用 `table()` 函数来创建一个频率表，并找出所有出现频率最高的值。这里有一个简单的函数定义示例：

```r
Mode <- function(x) {
  freq_table <- table(x)
  mode_values <- names(freq_table[freq_table == max(freq_table)])
  as.numeric(mode_values)
}

# 创建一个含有多个众数的向量
x <- c(1, 1, 2, 2, 3, 3, 4)

# 调用自定义的众数函数
mode_values <- Mode(x)
print(mode_values)

## 输出
# [1, 2, 3]
```

在这个函数中：

1. `table(x)` 计算向量 `x` 中每个值的出现频率。
2. `freq_table == max(freq_table)` 找出所有出现频率等于最大频率的值。
3. `names()` 获取这些频率值的名称，这些名称实际上就是向量 `x` 中的值。
4. `as.numeric()` 将结果转换为数值类型，因为 `table()` 函数的结果会返回因子类型的名称。

当你运行这个函数时，它会返回向量中所有的众数。如果有多个值出现频率相同且都是最高的，那么这些值都将被返回。


或直接在df上进行操作：

```r
# the function for computing mode
mode <- df_bbt %>% 
  group_by(Viewers..millions.) %>%
  summarise(count = n()) %>% 
  slice_max(count, with_ties = TRUE)

# the mean, median, and mode
cat(paste0("the mean of viewers:", round(mean(df_bbt$Viewers..millions.),2), "\n",
          "the median of viewers:", median(df_bbt$Viewers..millions.), "\n"))
cat("the mode of viewers:", mode$Viewers..millions.)
```