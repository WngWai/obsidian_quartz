plt.rcParams['font.sans-serif']=['SimHei']

这行代码是用来设置 Matplotlib 绘图时所**使用的字体为中文字体**。具体来说，`plt.rcParams` 是一个全局参数配置对象，它包含了很多参数，用来控制 Matplotlib 绘图时的各种设置。`font.sans-serif` 参数是其中一个字体相关的参数，它表示绘图时使用的无衬线字体，即中文字体。`['SimHei']` 是一个列表，其中包含了要使用的中文字体的名称，这里使用的是黑体。因此，这行代码的作用是设置 Matplotlib 绘图时所使用的字体为中文字体黑体（SimHei）。


plt.rcParams['axes.unicode_minus']=False

这行代码是用来解决 Matplotlib 绘图时**负号显示为方块**的问题。具体来说，`plt.rcParams` 是一个全局参数配置对象，它包含了很多参数，用来控制 Matplotlib 绘图时的各种设置。`axes.unicode_minus` 参数是其中一个控制负号显示的参数，它的默认值为 True，表示负号使用 Unicode 来显示，但是在某些情况下，Unicode 显示负号会出现方块的问题。因此，将 `axes.unicode_minus` 参数设置为 False，可以让 Matplotlib 绘图时使用正常的负号显示，而不是方块。