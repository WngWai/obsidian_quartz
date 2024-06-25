`WordCloud`是Python中一个用于生成词云图的类，它属于`wordcloud`库。词云是一种数据可视化技术，用于**展示文本数据中词语的频率**。在词云中，词语的大小通常与其频率成正比。

### WordCloud()类对象的参数
- `font_path`：用于指定**字体路径**，指定用于渲染文本的字体，用于显示词云中的文字。
解决中文显示不出来的问题！类似python软件一般不支持中文字体！
font_path='C:\Windows\Fonts\msyh.ttc'  **微软雅黑**
- `width`和`height`：词云画布的宽度和高度，默认为400x200像素。

- `max_words`：显示的最大词数，默认为200。
- `mask`：**形状遮罩**，一个NumPy数组，用于定义词云的形状。定义词云形状的遮罩，应为NumPy数组格式。如果指定了遮罩，生成的词云将采用遮罩的形状。
```python
from PIL import Image
mask_image = np.array(Image.open("词云遮罩图.png"))
```
- `background_color`：背景颜色，默认为黑色。

- **`margin`**: 词之间的间距，默认为2像素。
- `max_font_size`：最大字号，如果未指定，则使用图像高度 - 默认为None。
- `min_font_size`：在词云中显示的最小字号，默认为4号。

- `color_func`： 指定单词的颜色生成函数。如果未指定，将使用默认的颜色生成方案。
- `contour_width`和`contour_color`：用于绘制**遮罩轮廓的宽度和颜色**。**遮罩的外轮廓**！

**`stopwords`**: 停用词集合，即在生成词云时**要排除的词**。如果为None，使用默认的停用词集合。可以使用`set()`创建自己的停用词集合。

**`background_color`**: 背景颜色，默认为'black'。可以指定任何有效的颜色名称或代码。
**`mode`**: 当背景颜色为“None”时，模式为“RGBA”，并进行透明背景。否则为“RGB”。

11. **`relative_scaling`**: 字号调整逻辑。如果值为0，则只考虑词频排序；如果值为1，则字体大小与词频成线性关系，默认为0.5。


13. **`regexp`**: 使用正则表达式分割输入的文本，默认为`r"\w[\w']+"`，这意味着捕获字母数字字符和下划线的词。

14. **`collocations`**: 是否包括双词搭配（bigrams）。默认为True。

15. **`colormap`**: 用于生成词云中颜色的matplotlib颜色图名称，默认为“viridis”。可使用`matplotlib.pyplot.cm`中的任何颜色图。

17. **`contour_width`和`contour_color`**: 如果使用了`mask`，这两个参数用于绘制轮廓的宽度和颜色。

### 主要方法
- `generate(text)`：从文本字符串生成词云。这是最常用的方法之一，它分析文本并生成词云。
- `generate_from_frequencies(frequencies)`：从词频字典生成词云。这允许直接通过词频来生成词云，而不是从原始文本分析。
- `to_file(filename)`：将生成的词云保存为图片文件。
- `recolor(color_func=None, random_state=None)`：重新给词云上的词着色。可以通过定义`color_func`来自定义颜色方案。

### 使用介绍

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 定义一段文本
text = "Python is an amazing programming language. It is widely used in data science, web development, automation, and many other fields."

# 创建WordCloud对象
wordcloud = WordCloud()

# 用文本生成词云
wordcloud.generate(text)

# 使用matplotlib显示词云
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


### 如何指定展示的形状
图片的颜色通道不重要，因为数组将会被转换成一个**布尔**或灰度格式来表示遮罩的形状。通常，遮罩图片为黑白色效果最佳，其中白**色部分（或者说是亮部分）** 表示词云将**填充**的区域，**黑色部分（或暗部分）** 表示词云将会**留空**的区域。

![[Pasted image 20240302194455.png]]

![[Pasted image 20240302194502.png]]

是的，`wordcloud`库允许你自定义展示词云的形状。这主要通过`mask`参数实现，你可以利用这个参数来指定一个形状的遮罩图像。遮罩图像是一个NumPy数组，词云将填充这个形状的轮廓。遮罩图像中，通常背景部分是白色或者透明的，而形状部分是黑色或者其他有色部分。
如何使用`mask`自定义词云形状
1. **准备遮罩图像**：首先，你需要准备一个遮罩图像文件，这个文件定义了词云的形状。图像中的非白色部分将定义词云的形状。
2. **加载遮罩图像**：使用`PIL`或`matplotlib`等库读取遮罩图像，并转换为NumPy数组。
3. **生成词云**：在创建`WordCloud`对象时传入`mask`参数。

在这个示例中，`mask_image.png`是一个遮罩图像文件，它的非白色部分定义了词云的形状。`WordCloud`对象创建时通过`mask`参数接收这个遮罩图像（作为NumPy数组）。生成的词云将填满遮罩图像的形状部分。
