#### 1，df.fillna(value="missing", inplace=True).head()报错。
`df.fillna(value="missing", inplace=True)`会返回一个`None`对象，而不是一个`DataFrame`对象。这是因为`inplace=True`参数表示在原始数据上进行操作，而不是创建一个新的`DataFrame`对象。因此，如果你想在原始数据上进行填充缺失值的操作并返回一个`DataFrame`对象，你可以将`inplace=True`参数去掉，这样就会返回一个填充了缺失值的新的`DataFrame`对象，例如：

```python
df_filled = df.fillna(value="missing")
```

这样就会返回一个新的`DataFrame`对象`df_filled`，它是在原始数据`df`的基础上填充了缺失值的。


#### 2,关于数据中带有“’”单引号前缀的影响
**如果将使用单引号（'）前缀的数值数据**导入到 Pandas 的 DataFrame 中，这些数据在 DataFrame 中将被视为文本格式。也就是说，DataFrame 中的这些数据将被保留为字符串类型，而不是数值类型。在使用这些数据进行数值计算或分析时，需要使用字符串转换成数值类型。

例如，如果您希望对这些数据进行求和，可以使用以下代码将字符串转换成数值进行加法运算：

```python
df[column_name] = df[column_name].apply(lambda x: int(x))
df_sum = df[column_name].sum()
```

而如果将不使用单引号前缀的数值数据导入到 DataFrame 中，则这些数据将根据其格式被识别为数值类型。在进行数据分析时，不需要执行字符串到数值类型的转换，也不会出现数值计算或分析时的问题。


**如果一列数据同时包含了单引号（'）前缀和无单引号前缀的数值**，Pandas会将这列数据视为字符串类型，即使没有单引号前缀的数值本应该是数值类型。在进行数据分析时，这可能会导致错误的结果。因为字符串类型和数值类型之间的数学运算是不合法的。

因此，为了避免出现问题，我们在使用Pandas加载数据时，应该尽量避免出现单引号前缀和无单引号前缀混用的情况。如果一列数据应该是数值类型，那么应该将该列所有数据都设置为数值类型，即要么使用单引号前缀表示所有数据都是字符串类型，要么不使用单引号前缀表示所有数据都是数值类型。

如果您在加载数据时出现了单引号前缀和无单引号前缀混用的情况，可以使用Pandas中的类型转换方法将数据转换为正确的数据类型，如下所示：

``` python
# 将字符串类型转换为整数类型
df[column_name] = df[column_name].apply(lambda x: int(x.strip("'")))
``` 

或者
``` python
# 将字符串类型转换为浮点型
df[column_name] = df[column_name].apply(lambda x: float(x.strip("'")))
``` 

在以上代码中，`strip`方法用于去除字符串前面和后面的单引号。由于`apply`方法返回一个新Series，所以需要将转化后的Series重新赋值给该列。