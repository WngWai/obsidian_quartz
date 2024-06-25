装饰器是 Python 中用于在**不修改函数代码**的情况下**扩展或修改函数行为**的一种强大工具。它们本质上是**返回函数的函数**，允许你在函数调用之前或之后添加逻辑。

如何理解Python装饰器？ - 程序员志军的回答 - 知乎
https://www.zhihu.com/question/26930016/answer/99243411

装饰器本质上是一个Python函数，它可以让其他函数在不需要做任何代码变动的前提下增加额外功能，装饰器的返回值也是一个函数对象。它经常用于有**切面需求**的场景，比如：插入日志、性能测试、事务处理、缓存、权限校验等场景。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量与函数功能本身无关的雷同代码并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

@符号是装饰器的语法糖，在定义函数的时候使用，避免再一次赋值操作。

### 基本结构

一个基本的装饰器结构如下：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # 在函数调用前做一些事
        result = func(*args, **kwargs)
        # 在函数调用后做一些事
        return result
    return wrapper
```

使用装饰器：

```python
@my_decorator
def my_function():
    print("Hello, World!")

# 等同于 my_function = my_decorator(my_function)
my_function()
```

### 分类介绍
#### **函数装饰器**
这是最常见的装饰器，用于装饰普通函数。

   ```python
   def simple_decorator(func):
       def wrapper():
           print("Something is happening before the function is called.")
           func()
           print("Something is happening after the function is called.")
       return wrapper

   @simple_decorator
   def say_hello():
       print("Hello!")

   say_hello()
   ```

#### 带参数的装饰器
这种装饰器本身是一个返回装饰器的函数，允许你在装饰器中传递参数。

   ```python
   def repeat(num_times):
       def decorator_repeat(func):
           def wrapper(*args, **kwargs):
               for _ in range(num_times):
                   result = func(*args, **kwargs)
               return result
           return wrapper
       return decorator_repeat

   @repeat(num_times=3)
   def greet(name):
       print(f"Hello, {name}!")

   greet("Alice")
   ```

#### **类装饰器**
类装饰器用于装饰类，可以用于修改类的行为。

   ```python
   class DecoratorClass:
       def __init__(self, func):
           self.func = func

       def __call__(self, *args, **kwargs):
           print("Before the function call")
           result = self.func(*args, **kwargs)
           print("After the function call")
           return result

   @DecoratorClass
   def say_goodbye():
       print("Goodbye!")

   say_goodbye()
   ```

####  **方法装饰器**
用于装饰类的方法，通常用于检查或修改实例方法的行为。

   ```python
   def method_decorator(func):
       def wrapper(self, *args, **kwargs):
           print(f"Calling method {func.__name__}")
           return func(self, *args, **kwargs)
       return wrapper

   class MyClass:
       @method_decorator
       def method(self):
           print("This is a method")

   obj = MyClass()
   obj.method()
   ```

### 装饰器的实际作用
#### **日志记录**
装饰器常用于在函数调用前后添加日志记录，以便追踪函数调用。

   ```python
   def log_decorator(func):
       def wrapper(*args, **kwargs):
           print(f"Calling {func.__name__}")
           result = func(*args, **kwargs)
           print(f"Finished {func.__name__}")
           return result
       return wrapper

   @log_decorator
   def add(a, b):
       return a + b

   add(3, 4)
   ```

#### **性能计时**
可以用于测量函数执行时间，帮助优化代码性能。

   ```python
   import time

   def timing_decorator(func):
       def wrapper(*args, **kwargs):
           start_time = time.time()
           result = func(*args, **kwargs)
           end_time = time.time()
           print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
           return result
       return wrapper

   @timing_decorator
   def compute(x):
       time.sleep(x)
       return x

   compute(2)
   ```

#### **权限验证**
在函数调用前检查用户权限，确保只有授权用户可以执行特定操作。

   ```python
   def check_permission(user_role):
       def decorator(func):
           def wrapper(*args, **kwargs):
               if user_role != "admin":
                   raise PermissionError("You do not have permission to perform this action.")
               return func(*args, **kwargs)
           return wrapper
       return decorator

   @check_permission("admin")
   def delete_user(user_id):
       print(f"User {user_id} deleted")

   delete_user(123)
   ```

####  **缓存**
缓存函数的结果，以提高性能。

   ```python
   def cache_decorator(func):
       cache = {}
       def wrapper(*args):
           if args in cache:
               return cache[args]
           result = func(*args)
           cache[args] = result
           return result
       return wrapper

   @cache_decorator
   def slow_function(x):
       time.sleep(2)
       return x * 2

   print(slow_function(3))  # Takes 2 seconds
   print(slow_function(3))  # Returns immediately
   ```

### 总结

装饰器在 Python 中是一个强大且灵活的工具，用于在不修改函数或类本身的情况下增强其功能。通过装饰器，你可以添加日志、计时、权限检查、缓存等功能，使代码更简洁、更可读、更易维护。