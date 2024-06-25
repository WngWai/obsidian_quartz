`偏导数`：是一个**数**。
`梯度`：梯度的概念类似坐标点在高维空间中用向量指代，**与向量配套使用**，实质用于**高维空间的运算**。本质是一个**向量**（矢量），表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）。在单变量的实值函数的情况，**梯度只是导数**，或者，对于一个线性函数，也就是线的**斜率**。
`梯度下降算法`：梯度下降算法是一种**优化算法**，用于最小化或最大化函数。在深度学习中，主要使用梯度下降算法来最小化损失函数，以便优化神经网络的参数。梯度下降算法的基本思想是通过**迭代地更新参数**，沿着梯度的反方向**逐步调整参数**的整个过程，从而使损失函数逐渐减小。

layer神经元
[[tensor张量]]

**梯度**：
[[t.requires_grad_()]] 修改张量requires_grad属性，是否需要计算梯度，从而提前分配存储梯度的空间
[[t.backward()]] 反向求导，计算梯度值
[[t.retain_grad()]]保留**中间变量的梯度信息**
[[torch.no_grad()]]**阻止计算图记录**，从而避免新定义的张量被追踪
[[t.grad]] 获得t的**梯度值**
[[t.zero_()]] 对张量内所有元素进行清零

[[torch.save()]]存储训练模型
[[torch.load()]]加载保存好的训练模型
[[model.load_state_dict()]]承接的是上面torch.load得到的参数，详细后面再研究

[[updater.step()]]执行参数更新

### torch.cude
!nvidia-smi 查看**GPU信息**
[[torch.cuda.is_available()]]查看当前系统**是否支持CUDA加速**
torch.cuda.device_count()返回当前系统中**可用的 GPU 数量**，没有识别到或没有，返回值都是0
[[torch.device()]] 生成特定设备的torch.device类对象，方便其他函数中的device参数指定内容，动态管理设备和提高可读性
[[t.to()]] 将**张量转换到指定的设备上**进行计算

### torch.utils
提供了用于**数据加载和处理**的工具类和函数，方便地创建**数据集对象、数据加载器（DataLoader）和数据转换**等，以便在训练深度学习模型时进行数据的批量加载和预处理。utilities，即**工具函数或工具类**的集合

#### Dataset
[[Dataset数据集类]]

- 自定义

	[[data.Dataset]]基于**基类Dataset**构建**`自定义`数据集对象**，更加灵活，相对的是pytorch自带的训练数据集。类似R中的自定义任务。
	
	[[data.TensorDataset()]] 将数据和标签多张张量**封装成Dataset数据集对象**，也是自定义，适用比较**简单的情况**！

- 其他处理
	
	[[data.Subset()]]创建**子集对象**，从给定的数据集中选择特定的子集
	[[data.random_split()]]将一个数据集随机划分成多个子集的函数
	</br>
	[[data.get_worker_info()]]在多线程数据加载时获取当前工作线程的信息。
	[[data.Sampler()]]采样器类，用于定义数据集的采样策略

#### Dataloder
[[data.DataLoader()]] 根据Dataset对象，创建**数据加载器/迭代器**。从给定的训练集中每次加载生成小批次（量）的训练集

### torch.nn
torch.nn模块是PyTorch中用于**搭建神经网络**的库，提供了各种用于定义神经网络层、损失函数和优化器的类和函数。它包含了各种预定义的神经网络模块，如线性层、卷积层、循环神经网络层等，方便用户构建自定义的神经网络模型。

#### 模型容器（Model Containers）：
实现自定义神经网络框架，而之前的相当于库自带的！
[[nn.Module类型]]**模块类**，定义更加**复杂**的**神经网络**模型，所有神经网络模块的基类，我们可以通过继承这个类来定义自己的神经网络模型，可以嵌套入nn.Sequential()
https://blog.csdn.net/zerone_zjp/article/details/108624224

[[nn.Sequential()]] **序列模型**（Sequential Model），简单的**线性堆叠结构**，堆叠模块的容器，将每层神经元进行组合，进行简单的定义

[多种容器的比较](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%94%E7%AB%A0/5.1%20PyTorch%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89%E7%9A%84%E6%96%B9%E5%BC%8F.html)
[[nn.ModuleList()]]保存和管理子模块的容器
[[nn.ModuleDict()]]用于保存和管理子模块的容器

#### 张量处理
[[nn.Flatten()]]将多维输入张量展**平为一维张量**，在用softmax回归处理图片输出时会用到
[[nn.Parameter()]]将张量包装成**可训练的参数** ？？？实际应用举例更新

#### 网络层（Layers）
全连接层：
[[nn.Linear()]] 本身是神经网络的**线性变换层**（也称全连接层），因为线性回归模型相当于单层神经网络，可以直接当作线性回归模型。

一、二、三维卷积层：
[[nn.Conv2d()]]2D卷积层
一、二、三维池化层：
[[nn.MaxPool2d()]]2D最大池化层
[[nn.AvgPool2d()]]2D平均池化层

[[nn.BatchNorm2d()]]2D批归一化层
[[nn.Dropout()]]用于模型正则化

#### 损失函数（Loss Functions）：
[[nn.MSELoss()]]**均方误差**损失函数MEM，又称**平方L2范数**（L2 Loss）
[[nn.BCELoss()]] **二元交叉熵损失**函数，用于二分类问题。
[[nn.CrossEntropyLoss()]]**交叉熵损失函数**，用于多类别分类问题。
[[nn.NLLLoss()]] **负对数似然损失函数**，用于多类别分类问题。


#### 激活函数（Activation Functions）：

[[nn.ReLU()]]: ReLU激活函数。

[[nn.Softmax()]]: Softmax激活函数，用于多类别分类问题。

[[nn.Sigmoid()]]: Sigmoid激活函数。

`Tanh`: 双曲正切激活函数。

#### 初始化（Initialization）nn.init：
后缀带有下划线，直接原地更改输入张量的值。

[[init_weights()]]**自定义初始化**，根据不同类型层，设定不同的权值初始化方法，这种比直接对指定层进行初始化要精细、简洁

[[torch.nn.init.uniform_()]]: 从**均匀分布**中初始化权重。

[[torch.nn.init.normal_()]]: 从`正态分布`中初始化权重。

torch.nn.init.constant_: 使用**常数**初始化权重。

torch.nn.init.zeros_: 将权重初始化为`零`。

torch.nn.init.ones_: 将权重初始化为**1**。

torch.nn.init.eye_: 将权重初始化为**单位矩阵**。

torch.nn.init.orthogonal_: 使用**正交矩阵**初始化权重。

torch.nn.init.sparse_: 使用**稀疏矩阵**初始化权重。

- xavier（Glorot）初始化:

	[[nn.init.xavier_uniform_()]]Xavier/Glorot**均匀**初始化方法初始化权重

	torch.nn.init.xavier_normal_: 使用 Xavier **正态分布**初始化权重。

- kaiming（He）初始化:

	[[nn.init.kaiming_uniform_()]]Kaiming**均匀**初始化方法初始化权重

	torch.nn.init.kaiming_normal_: 使用 Kaiming **正态分布**初始化权重。

#### 循环神经网络（Recurrent Neural Networks）：
现成的神经网络：
`RNN`、`LSTM`、`GRU`: 分别对应普通循环神经网络、长短期记忆网络、门控循环单元。

### torch.optim定义优化器
对于线性回归模型来说，就是执行梯度下降，设置初始化参数、学习效率，逐步优化参数的过程！
提供了各种**优化算法**的实现，用于优化模型的参数，更新神经网络的权重。它包含了常见的**优化器**，如随机梯度下降（SGD）、Adam、Adagrad等，可以帮助用户方便地在训练过程中更新模型参数。

[[torch.optim.optimizer]]**优化器的基类**。每个优化器都是一个类，我们一定要进行实例化才能使用
```python
class Net(nn.Module):
    ···
net = Net()
optim = torch.optim.SGD(net.parameters(),lr=lr) # 定义优化器
optim.step() # 更新模型参数
```
#### 优化器（Optimizers）：
[[optim.SGD()]] 随机**梯度下降**优化器,随机梯度递减，优化参数w，b
[[optim.Adam()]]**Adam优化器**，比较平滑的SGD，对学习率没有那么敏感，可以适应更宽的学习率
AdamW: 带有权重衰减的Adam优化器。
[[optim.RMSprop()]]RMSprop优化器。
[[optim.Adagrad()]]Adagrad优化器。
[[optim.Adadelta()]]Adadelta优化器。

#### 学习率调度器（Learning Rate Schedulers）：
[[optim.StepLR()]]学习率按步长衰减
[[optim.MultiStepLR()]]学习率按步长衰减。
[[optim.ExponentialLR()]]学习率按指数衰减。
[[optim.ReduceLROnPlateau()]]根据验证集表现自动调整学习率。
[[optim.CosineAnnealingLR()]] 学习率按余弦退火衰减。


torch.multiprocessing 多进程

