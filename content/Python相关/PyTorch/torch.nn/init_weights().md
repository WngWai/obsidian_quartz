人们常常将各**种初始化方法**定义为一个`initialize_weights()`的函数并在模型初始后进行使用。

```python
def initialize_weights(model):
	for m in model.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.zeros_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()	
```

这段代码流程是遍历当前模型的每一层，然后判断各层属于什么类型，然后**根据不同类型层，设定不同的权值初始化方法**。我们可以通过下面的例程进行一个简短的演示：

```python
# 模型的定义
class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Conv2d(1,1,3)
    self.act = nn.ReLU()
    self.output = nn.Linear(10,1)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)

mlp = MLP()
print(mlp.hidden.weight.data)
print("-------初始化-------")

mlp.apply(initialize_weights)
# 或者initialize_weights(mlp)
print(mlp.hidden.weight.data)

tensor([[[[ 0.3069, -0.1865,  0.0182],
          [ 0.2475,  0.3330,  0.1352],
          [-0.0247, -0.0786,  0.1278]]]])
"-------初始化-------"
tensor([[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]]])
```

**注意：** 我们在初始化时，最好不要将模型的参数初始化为0，因为这样会导致**梯度消失**，从而影响模型的训练效果。因此，我们在初始化时，可以使用其他初始化方法或者将模型初始化为一个很小的值，如0.01，0.1等。