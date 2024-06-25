在Python中，将脚本打包成exe文件是一个常见的需求，尤其是当你需要在没有安装Python环境的计算机上运行脚本时。目前，有几种流行的工具可以帮助你完成这个任务，最常用的有PyInstaller、cx_Freeze和py2exe。这里将简要介绍如何使用PyInstaller来打包Python脚本为exe文件，因为它支持多个平台（Windows、Linux和Mac OS X）并且易于使用。

【Python程序打包成exe文件的小技巧 py生成exe 脚本打包 可执行程序 Python脚本转换exe程序】https://www.bilibili.com/video/BV16h411Y7Yc?vd_source=6ba5a7ec009a0f45fd393fcd989921f7

如何使用pyinstaller打包python脚本？ - 玉米子禾的回答 - 知乎
https://www.zhihu.com/question/52660083/answer/3240674484

2个技巧，学会Pyinstaller打包的高级用法 - 程序员云岫的文章 - 知乎
https://zhuanlan.zhihu.com/p/398619997

Pyinstaller 打包Python脚本踩坑之旅 - young的文章 - 知乎
https://zhuanlan.zhihu.com/p/141853045
应该用虚拟环境，上面这篇文章没有用！


### 使用PyInstaller打包Python脚本为exe

1. **安装PyInstaller**:
   在命令行中使用pip安装PyInstaller：
   
   ```shell
   pip install pyinstaller
   ```

2. **打包脚本**:
   在命令行中运行PyInstaller，指定你的脚本名称。以下是一些常用的参数：

顺带说一句pyinstaller打包体积大，运行速度慢的问题，是因为好多不相干的包被一起打包进exe了，可以做个虚拟环境解决，pipenv就挺好用。

   - `-F` 或 `--onefile` : 打包成一个单独的exe文件。
   - `-w` 或 `--windowed` : 不显示命令行窗口（通常用于GUI应用）。
   - `-i` : 指定一个图标文件（.ico）给你的exe文件。
   
   例如，如果你有一个名为`script.py`的脚本，想要打包成单个文件，你可以运行：
   
   ```shell
   pyinstaller --onefile script.py
   ```
   
   如果你的脚本是一个GUI应用，不希望有命令行窗口出现，可以加上`-w`标志：
   
   ```shell
   pyinstaller --onefile --windowed script.py
   ```
   
   如果你想添加图标：
   
   ```shell
   pyinstaller --onefile --windowed --icon=app.ico script.py
   ```

3. **查找生成的exe文件**:
   PyInstaller会在`dist`目录下生成exe文件，你可以在这个目录中找到你的exe文件。如果你使用了`--onefile`选项，应该只有一个exe文件。

4. **测试exe文件**:
   双击运行`dist`目录下的exe文件，检查你的程序是否按预期运行。

5. **打包其他资源**:
   如果你的应用还依赖其他文件（如图片、数据文件等），需要确保这些资源正确打包。PyInstaller通常可以自动找到并包括这些资源，但有时你可能需要手动指定它们。

6. **分发exe文件**:
   你可以将`dist`目录下的exe文件分发给其他用户，他们无需安装Python环境就可以运行。

注意：编译成exe后的程序可能比原始的Python脚本大很多，因为它包括了一个Python解释器和你的脚本使用的所有依赖库。此外，使用PyInstaller等工具打包的应用程序在第一次启动时可能启动较慢，因为它需要解压缩到一个临时位置。