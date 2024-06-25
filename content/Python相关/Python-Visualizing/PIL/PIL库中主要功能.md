Python中的PIL库（现在被Pillow代替，作为PIL的友好分支）的`Image`模块提供了一系列的功能来处理图像。以下是根据功能分类的主要函数介绍：

### 1. 打开和保存图片

- **open(fp, mode='r')**: 打开并识别给定的图像文件。
- **save(fp, format=None, **params)**: 保存当前图片到指定文件。

### 2. 图像转换

- **convert(mode=None, matrix=None, dither=None, palette=0, colors=256)**: 转换图像的颜色模式。
- **resize(size, resample=0)**: 调整图像大小。
- **rotate(angle, resample=0, expand=0, center=None, translate=None, fillcolor=None)**: 旋转图像。
- **transpose(method)**: 翻转或旋转图像。
- **crop(box=None)**: 裁剪图像。
- **thumbnail(size, resample=3)**: 更改图像的大小，使其适合指定的大小。

### 3. 图像增强与滤镜

- **filter(filter)**: 对图像应用滤镜。
- **point(lut, mode=None)**: 对图像的每个像素应用一个给定的转换。
- **enhance(factor)**: 用于色彩、对比度、亮度、锐度等的增强。

### 4. 绘图和文本

- **paste(im, box=None, mask=None)**: 将一个图像粘贴到另一个图像上。
- **draw()**: 创建一个可以在图像上绘制的对象。

### 5. 图像信息

- **getdata(band=None)**: 返回图像的内容作为对象。
- **getbbox()**: 计算图像的边界盒。
- **getcolors(maxcolors=256)**: 获取图像中的颜色。
- **getextrema()**: 获取图像中每个通道的最大和最小像素值。
- **getpixel(xy)**: 获取指定位置的像素值。
- **histogram(mask=None, extrema=None)**: 计算图像的直方图。

### 6. 图像序列（针对GIF等动态图）

- **seek(frame)**: 移动到序列文件中的指定帧。
- **tell()**: 返回当前帧的序号。
- **n_frames**: GIF图像的帧数。
- **is_animated**: 图像是否为动画。

### 7. 其他

- **split()**: 分离图像的通道。
- **merge(mode, bands)**: 合并不同的通道以创建图像。
- **mode**: 图像的颜色模式，如"RGB"。
- **size**: 图像的尺寸，以像素为单位。

这些函数大致涵盖了Pillow库中Image模块的主要功能。不过，请注意，Pillow库提供的功能远远不止这些，具体的使用方法和更多的功能可以通过查阅官方文档来获取。