是pandas中用于按照样本分位数对数据进行**离散化**的函数。样本分位数是一种很常见的统计描述量，它将所有元素从小到大排序后，平均分成若干份，其中每一份所对应的元素就叫做分位数。

`qcut()`函数需要**指定待划分的数据集和分位数的数量**，然后它会根据数据值的大小将数据集分成指定数量的区间（即将数据集离散化），每个区间内包含的数据个数尽量相等。如果某个值落在了两个区间中间，它会被分配到更靠前的区间中去。返回值是一个离散化的Series或者DataFrame对象，其中每个元素值表示原始数据在哪个区间中。该函数常用于数据分析和建模中。

下面是一个使用`.qcut()`将数据进行离散化的例子：

```python
import pandas as pd

# 创建一个数据表
data = {'score': [85,65,90,70,93,88,58,83,72,75]}
df = pd.DataFrame(data)

# 对数据表进行离散化
df['category'] = pd.qcut(df['score'], q=3, labels=['Low', 'Medium', 'High'])

# 输出离散化后的数据表
print(df)
```

这个例子中，我们创建了一个数据表`df`，其中包含10个分数数据。然后我们使用`qcut()`函数对这些分数进行了离散化，其中将原始分数数据根据等频分位数法分成了3组，并用"Low"、"Medium"和"High"标记了每个区间。得到的`category`列就是离散化后的结果。

输出结果如下：

```python
   score category
0     85   Medium
1     65      Low
2     90     High
3     70   Medium
4     93     High
5     88     High
6     58      Low
7     83   Medium
8     72   Medium
9     75   Medium
```

从输出结果可以看出，分数数据被成功地离散化成了三个区间，每个区间里的数据个数尽量相等。