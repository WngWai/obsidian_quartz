## 数据的基础处理
### 筛选行/列
[filter()](R相关/R/dplyr/filter().md) 处理数据框数据（df），筛选满足条件的**行**

filter_if()选择满足**特定条件的行**,跟filter()的差别是？

[[slice()]]**选取或排除**指定位置的行

[select()](select().md) 按照**指定列**，形成新的数据框

[select_if()](select_if().md)选择满足**特定条件的列**

### 对行/列数据进行操作
[arrange()](arrange().md) 对df中行数据按**指定列中数据进行重新排序**

[order()](order().md) 返回排序的**索引值**，内置排序函数

[min_rank()](min_rank().md) 最小值**排序序号**

[sort()](sort().md) 对元素进行排序，内置函数


[mutate()](mutate().md) 根据旧列添加新列，或者替换旧列数，配上[across()](across().md)在多个列上应用相同的变换或统计函数

[mutate_if()](mutate_if().md)**满足特定条件的列上应用相关函数**


[transmute()](transmute().md) 只**保留新列**


[lag()](lag().md) 和[lead()](lead().md)**偏移**函数，后移和前移

## 分组统计
[group_by()](group_by().md) 指定列进行**分组**，分组后再用summarize()后会**保留**分组列

[cut()](cut().md)函数，进行固定范围值添加新组

[ungroup()](ungroup().md) **取消**分组，在使用管道符进行参数传递中使用的是同一个源数据，所以要及时撤销分组操作！

- 聚类

	[summarize()、summarise()](summarize()、summarise().md) **统计分析**列数据

	[slice_max()](slice_max().md)指定最大值的观测行

	slice_min()


[n()](n().md) 计算**行数**

[count()](count().md)计算**唯一值**出现**次数**，跟python不同

[n_distinct()](n_distinct().md)计算**种类数**，跟上面的唯一值次数指频数不同

## 处理关系数据

[inner_join()](inner_join().md) 内连接

[left_join()](left_join().md)左连接

right_join()右连接

[full_join()](full_join().md) 全连接

[semi_join()](semi_join().md)半连接，目的筛选**左表数据**，**类似交集但只保留做表数据**。以右表数据作为标准，筛选左表中存在于右表中的数据，并**不会返回右表中任何数据**。右有左也有的数据。

[anti_join()](anti_join().md)反连接，目的是筛选左表数据，跟**半连接相反**，**筛选右表没有的数据**，返回在第一个数据框中存在而在第二个数据框中不存在的行。右无，左有的数据。

[merge()](merge().md) 内置函数，不建议用

