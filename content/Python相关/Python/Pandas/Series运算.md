Series 运算通常指的是对一组数据进行各种数学运算，包括加、减、乘、除、求和、平均值等。在 Python 中，可以使用 Pandas 库中的 Series 对象来进行这些运算。

下面是一些示例：

1. 创建一个 Series 对象
```python
import pandas as pd  
data = [1, 2, 3, 4, 5] 
s = pd.Series(data)
```

2. 加法运算(减法同理)
根据**索引相加**的，如果两个Series对象在某个索引位置上的值都存在，则它们在该位置上的值将相加，否则缺失值**将被视为0**来进行计算。
```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
s3 = s1 + s2
print(s3)
```
输出：
```
a    NaN
b    6.0
c    8.0
d    NaN
dtype: float64
```

3. 减法运算
```python
s1 = pd.Series([10, 20, 30]) 
s2 = pd.Series([5, 10, 15]) 
s3 = s1 - s2
```

4.除法运算（乘法同理）
两个Series对象相除，会根据它们的**索引进行对应相除**。如果两个Series对象的索引不完全一致，Pandas会自动进行对齐操作，在对齐时缺失的值会被填充为NaN，然后再进行相除计算
```python
import pandas as pd

s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])

s3 = s1 / s2
print(s3)
```
输出：
```
a   NaN 
b   0.5 
c   0.5 
d   NaN 
dtype: float64`
```
5. 数乘运算（“数除”就是乘分数）
```python
s1 = pd.Series([1, 2, 3]) 
k = 10 
s2 = k * s1
```
6. 求和运算
```python
s1 = pd.Series([1, 2, 3]) 
sum_s1 = s1.sum()
```
7. 求平均值运算
```python
s1 = pd.Series([10,20,30]) 
mean_s1= s.mean()
```
还有很多其他的 Series 运算，具体可以参考 Pandas 文档。