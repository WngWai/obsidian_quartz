在 `openpyxl` 包中，并没有直接的 `save()` 函数。在 `openpyxl` 中，保存工作簿的方法是使用 `save()` 方法，该方法属于 `Workbook` 对象。以下是相关的介绍和详细示例：

```python
workbook.save(filename)
```


- **filename** (str): 要保存的文件名，可以包含路径。

该方法没有返回值。

### 详细举例：

```python
from openpyxl import Workbook

# 创建一个新的工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 在工作表中写入数据
sheet['A1'] = 'Hello'
sheet['B1'] = 'World'

# 保存工作簿到文件
workbook.save('example.xlsx')
```

在这个例子中，我们首先使用 `Workbook()` 创建一个新的工作簿，然后获取默认的工作表。接着，在工作表中写入一些数据。最后，使用 `save()` 方法将工作簿保存到名为 'example.xlsx' 的文件中。

请注意，`save()` 方法接受一个参数，即保存文件的路径。如果文件已经存在，将被覆盖。如果路径中包含目录，确保目录已经存在。

`openpyxl` 还提供了其他一些保存选项，例如设置文件的写入方式、设置密码等，但这些选项是可选的，不是必需的。上述示例中的简单用法是最基本的保存工作簿的方式。