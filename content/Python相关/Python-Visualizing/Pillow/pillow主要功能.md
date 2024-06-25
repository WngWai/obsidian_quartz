Pillow是Python的一个图像处理库，它提供了广泛的文件格式支持、有效的内部表示和相当强大的图像处理能力。

### 1. 打开和保存图像

- **open(fp, mode='r')**: 打开图像文件。
- **save(fp, format=None, **params)**: 保存图像到文件。

### 2. 图像信息和属性

- **format**: 图像的格式（例如，JPEG，PNG）。
- **size**: 图像的尺寸（宽度，高度）。
- **mode**: 图像的模式（例如，RGB，CMYK）。
- **info**: 图像相关信息的字典（例如，exif数据）。

### 3. 图像转换

- **convert(mode, **kwargs)**: 转换图像到不同的颜色模式。
- **thumbnail(size, resample=Image.LANCZOS)**: 创建图像缩略图。

### 4. 图像处理

- **crop(box=None)**: 裁剪图像。
- **rotate(angle, resample=Image.NEAREST, expand=0, center=None, translate=None, fillcolor=None)**: 旋转图像。
- **resize(size, resample=Image.NEAREST)**: 调整图像大小。
- **transpose(method)** 或 **transpose(Image.FLIP_LEFT_RIGHT)**: 镜像或翻转图像。

### 5. 图像滤镜和效果

- **filter(filter)**: 应用预定义滤镜。
- **ImageFilter模块**: 提供各种滤镜，如模糊，边缘增强等。

### 6. 图像绘制和图形

- **ImageDraw模块**: 提供简单的2D图形绘制。
  - **Draw(image, mode=None)**: 创建一个可以在给定图像上绘制的对象。

### 7. 图像序列（例如GIF）

- **seek(frame)**: 移动到序列中的某一帧。
- **tell()**: 返回当前帧的帧号。
- **n_frames**: 序列中的帧数。

### 8. 颜色空间和色彩管理

- **ImageCms模块**: 颜色管理和转换功能。

### 9. 其他工具和实用程序

- **ImageChops模块**: 提供一些简单的通道操作函数，如加法、乘法等。
- **merge(mode, bands)**: 把几个通道合并成一个图像。
- **split()**: 把图像分割成单独的通道。

这只是Pillow功能的一个概述，更多细节和功能可以在Pillow的官方文档中找到。由于Pillow是一个不断发展的库，建议查阅最新版本的官方文档以获取最完整的信息。