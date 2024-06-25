python是**面向对象**的语言，对比面向过程的语言，其实两者类似清单和定额的关系。**清单由定额组成，这样简化了计量工作**。

类的思路可以**类比图纸对于建筑**，用绘制好的图纸批量生产类似建筑。用事先定义好的类，创建同类型的对象。

进级！！！
![[Pasted image 20240129211149.png]]


```python
class Dog:  
	dogbook = {'黄色':30, '黑色':20, '白色':0} # 类属性：所有类的实例都有该属性
    name = None  # 实例属性：没用进行赋值，对成员变量进行了声明，可以不用声明  

# 构造方法，包含于魔术方法，包含于实例方法。
def __init__(self, name, color, weight):  
	self.name = name  # 这也是实例属性。将局部变量name(习惯性将局部变量命名与实例变量名相同)的值赋给了实例变量self.name  
	self.color = color  
	self.weight = weight  
#此处省略若干行，应该更新dogbook的数量  
  
#实例方法: 定义时,必须把self作为第一个参数，可以访问实例变量，只能通过实例名访问  
def bark(self):  
	print(f'{self.name} 叫了起来')  
  
#类方法：定义时,必须把类作为第一个参数，可以访问类变量，可以通过实例名或类名访问  
@classmethod  
def dog_num(cls):  
	num = 0  
	for v in cls.dogbook.values():  
		num = num + v  
	return num  
  
#静态方法：不强制传入self或者cls,  一般在类中使用？
@staticmethod  
def total_weights(dogs):  
	total = 0  
	for o in dogs:  
	    total = total + o.weight  
	return total  
  
print(f'共有 {Dog.dog_num()} 条狗')  
d1 = Dog('大黄', '黄色', 10)  
d1.bark()  
print(f'共有 {d1.dog_num()} 条狗')  
  
d2 = Dog('旺财', '黑色', 8)  
d2.bark()  

print(f'狗共重 {Dog.total_weights([d1, d2])} 公斤')
```

## 实例方法（Instance Methods）
第一个参数为self，为实例对象本身，实例方法**绑定到对象实例**，在实例上进行调用，也可以在调用时传递额外参数，对实例变量进行操作，self.* 。定义在类中的方法，用于**操作类的实例**（对象）。创建类的实例后，实例方法可以在该实例上被调用，并且可以访问和操作实例的属性。
`依据实例调用实例方法，操作实例属性和传入的参数！`

`实例化对象`：创建类的实例（对象）的过程。通过调用类名后面跟着括号，可以创建类的实例。

跟对象self绑定的好处是能**调用对象的属性**！
![[Pasted image 20231221084114.png]]

方法也能**传入参数**，实现函数的功能，只是更强调与实例属性的联动
![[Pasted image 20231221084943.png]]


```python
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def display_value(self):
        print("Value:", self.value)
    
    def increment_value(self, increment):
        self.value += increment

# 创建类的实例
obj = MyClass(10)

# 调用实例方法
obj.display_value()  # 输出: Value: 10

obj.increment_value(5)
obj.display_value()  # 输出: Value: 15
```


### 魔术方法
`魔术方法`（Magic Methods）：在类中使用特定命名的特殊方法。执行类的特定操作或实现特定的行为，当**满足特定的条件**时，Python解释器会**自动调用**这些方法。

1. `__init__(self, ...)`：也称构造方法，初始化方法，在创建对象时进行初始化操作。
`构造方法(特殊的实例方法)`:构造方法在**创建类的实例对象时自动调用**，并允许传**递参数来初始化对象的属性**，说人话就是定义类的实例对象时直接进行**实例属性赋值**。用__init__具有两个特点，构建类时传入的参数会自动提供给__init__方法；构建类时__init__方法会强制执行；
```python
class MyClass:
    def __init__(self, name):
        self.name = name

obj = MyClass("Alice")
```

2. `__str__(self)`：返回**对象的字符串表示**。
在使用print()函数输出对象时，会执行str魔术方法。**打印对象，从而得到统一的输出内容！**
```python
class MyClass:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return f"MyClass object with name: {self.name}"

obj = MyClass("Alice")
print(obj)  # 输出: MyClass object with name: Alice
```

3. `__len__(self)`：返回**对象的长度**。
```python
class MyList:
    def __init__(self):
        self.data = []
    
    def __len__(self):
        return len(self.data)

my_list = MyList()
my_list.data = [1, 2, 3, 4, 5]
print(len(my_list))  # 输出: 5
```

4. `__getitem__(self, key)`：通过索引或键获取对象的元素。
```python
class MyList:
    def __init__(self):
        self.data = []
    
    def __getitem__(self, index):
        return self.data[index]

my_list = MyList()
my_list.data = [1, 2, 3, 4, 5]
print(my_list[2])  # 输出: 3
```

5. `__setitem__(self, key, value)`：通过索引或键设置对象的元素。
```python
class MyList:
    def __init__(self):
        self.data = []
    
    def __setitem__(self, index, value):
        self.data[index] = value

my_list = MyList()
my_list.data = [1, 2, 3, 4, 5]
my_list[2] = 10
print(my_list.data)  # 输出: [1, 2, 10, 4, 5]
```


## 类方法
第一个参数是cls，类方法**绑定到类**，通过**类名（不需要实例化对象也可调用）或实例名**都可以调用，调用时可传参数，可以对**类变量cls.*** 进行操作，或者**传入的参数**进行操作。
使用`@classmethod`装饰器，它将方法标记为类方法，并将第一个参数命名为`cls`，该参数**引用类本身**。
`不必实例化也可调用类方法，并且操作类属性或传入的参数，不可访问实例属性`
```python
class MyClass:
    class_variable = "class value"

    @classmethod
    def class_method(cls):
        print(f"Called class_method of {cls}")
        print(f"Class variable is {cls.class_variable}")

MyClass.class_method()  # 通过类调用
obj = MyClass()
obj.class_method()  # 也可以通过实例调用


class MathUtils:
    @classmethod
    def add(cls, a, b):
        return a + b

result = MathUtils.add(5, 3)
print(result)  # 输出: 8
```


## 静态方法
没有cls和self参数，静态方法**不绑定到类或对象实例**，实际调用时可以用**类名或实例名**，与类有关，实际上可以理解为类外的函数，只是跟类紧密相关。用@staticmethod装饰器，将方法标记为静态方法。
`像一个独立的函数，不依赖类和实例对象，但在类和实例中又会被额外使用到！`
```python
class MyClass:
    @staticmethod
    def static_method():
        print("Called static_method")

MyClass.static_method()  # 通过类调用
obj = MyClass()
obj.static_method()  # 也可以通过实例调用
```


