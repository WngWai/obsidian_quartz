是`readr`包中的一个函数，用于**读取逗号分隔**的文本文件（CSV格式）。它是`readr`包中最常用的函数之一，具有高效和快速读取大型数据集的能力。

read.csv()是R语言中的基础包中函数，而非readr包中read_csv()的！

```R
read_csv(file, col_names = TRUE, col_types = NULL, skip = 0, na = c("", "NA"), ...)
```

参数说明：

- `file`：要读取的CSV文件的路径或URL。

* `sep`：设置数据分隔符，默认应该是**空格**？

- `col_names`：一个逻辑值，指示**是否将文件的第一行作为列名**。默认为`TRUE`。

	也可以指定列名，col_names = c('')

- `col_types`：一个列类型的字符向量，用于指定**每列的数据类型**。默认为`NULL`，表示自动推断数据类型。

- `skip`：要**跳过的行数**。默认为0，表示不跳过任何行。

	1， 跳过第一行，以便正确识别列标题。

- `na`：一个字符向量，用于指定要解析为缺失值的字符串。默认为`c("", "NA")`，表示**空字符串和"NA"被解析为缺失值**。

- col_select 读取指定列

	通过**列名**选择：col_select = "column_name"，col_select = c("column1", "column2")
	
	通过**列索引**选择：col_select = column_index，col_select = c(column_index1, column_index2)
	
	通过逻**辑向量**选择：col_select = c(TRUE, FALSE, TRUE)

- `...`：其他参数，用于进一步控制数据读取过程，如`locale`、`comment`等。

	**show_col_types = FALSE** 静默相关信息


```R
library(readr)

data <- read_csv("data.csv")
```


### 常见错误
读取CSV文件并存储为数据框
data <- read.csv("your_file.csv")

如果CSV文件使用了分号作为分隔符，可以指定sep参数
data <- read.csv("your_file.csv", sep = ";")


