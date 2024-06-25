
### 创建ndarray
[[arr.array()]]  
[[ndarray存储类型]]
[[arr.asarray()]] 一般用np.array，特殊下用此函数

[[arr.zeros()]]生成0数组，先占位
[[arr.ones()]]生成1数组，先占位
[[arr.linspace()]]，一维等间距数组
[[arr.arange()]]，一维等间隔的数组
[[arr.random.uniform()]]，均匀分布的随机数组
[[arr.random.normal()]]，正太分布的随机数组

### 生成新ndarray

[[arr.copy()]]  arr.copy()也行

### 属性
[[arr.shape]] 形状
arr.ndim 维度
arr.size 元素数量
arr.dtype 类型
arr.itemsize 元素字节大小
arr.T 转置

### 索引、切片
arr[x, y, z] 索引
arr[x, y, :] 切片

### 增
[[arr.hstack()]] 水平拼接，注意行数里是元组形式
[[arr.vstack()]] 垂直拼接
[[arr.concatenate()]] 可指定行、列连接，默认行
[[arr.split()]] 拆分

### 删


### 改
#### 改形状
[[arr.reshape()]] 没有动nd，返回新数组，将源数组根据形状只是根据形状重新排布返回
[[arr.resize()]] 动了nd，效果同reshape()
#### 修改类型
[[arr.astype()]]返回新创建的数组，原数组不改变
[[arr.tostring()]]跟传统的字符串有些差别，用的少
[[arr.tolist()]]将**数组转换为列表**

### 查

## 运算
### 逻辑运算
运算符
arr[nd > 1] 直接进行判断得到布尔值，显示布尔值为真的数值
print(nd > 1) 可以直接输出布尔值

通用判断函数
[[arr.all()]] 判断数组中所有元素是否存在满足条件的情况，或着某行/列所有数据是否满足某种情况
[[arr.any()]] 针对任意，即存在满足条件的元素就为真

三元运算符
[[arr.where()]] 根据指定条件，输出值或索引
[[arr.logical_and()]] 对两数组的元素实行逻辑且的操作
[[arr.logical_or()]] 对两数组的元素实行逻辑或的操作
### 统计运算
[[arr.max()]] axis=0找每列最大值，axis=1找每行最大值 
[[arr.argmax()]] 同上，但找最大值所在的位置
### 数组间运算
满足(m\*n)\*(n\*l)=(m\*l)要求
[[arr.matmul()]] 针对矩阵乘法
[[arr.dot()]] 点积操作
mat1\*mat2 直接乘，针对matrix矩阵对象


