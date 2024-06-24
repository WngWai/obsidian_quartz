用于**对新数据进行预测**，得到预测结果。

  ```r
  compute(x, covariate)
  ```
  
  - **x**：一个神经网络模型对象（由 `neuralnet` 函数生成）。
  - **covariate**：新数据。

## 参数介绍
predictions <-  compute(nn, covariate)
print(predicitons)
```python
Formula: ~.
10 variables, 0 factors
Variables and levels will be separated by '.'
A less than full rank encoding is used
Show in New Window
$neurons
$neurons[[1]]
       x1 x2
[1,] 1  1  0
[2,] 1  0  1

$neurons[[2]]
     [,1]      [,2]      [,3]      [,4]
[1,]    1 0.2331421 0.5497835 0.3091396
[2,]    1 0.6452519 0.7998846 0.3484325


# 实际预测结果
$net.result
          [,1]
[1,] 0.4361072
[2,] 0.5530290
```