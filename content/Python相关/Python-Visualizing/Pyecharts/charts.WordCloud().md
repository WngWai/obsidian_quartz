在Python的Pyecharts库中，`WordCloud()`函数用于创建词云图（Word Cloud）。
**函数定义**：
```python
pyecharts.charts.WordCloud()
```

**参数**：
以下是`WordCloud()`函数中常用的参数：

- `init_opts`（可选）：初始化选项，用于配置图表的基本属性，如标题、背景颜色等。默认值为`opts.InitOpts()`。

- `page_title`（可选）：图表所在HTML页面的标题。默认值为`"Word Cloud"`。

- `width`（可选）：图表的宽度。可以是像素值（如`"800px"`）或百分比（如`"80%"`）。默认值为`"800px"`。

- `height`（可选）：图表的高度。可以是像素值（如`"600px"`）或百分比（如`"60%"`）。默认值为`"600px"`。

**示例**：
以下是使用`WordCloud()`函数创建词云图的示例：

```python
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

# 定义词语和词频
words = [
    ("Hello", 100),
    ("World", 80),
    ("Python", 60),
    ("Echarts", 40),
    ("Data", 20),
    ("Visualization", 10),
]

# 创建词云图
wordcloud = (
    WordCloud()
    .add(series_name="热点分析", data_pair=words, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="词云示例"))
)

# 渲染图表到HTML文件
wordcloud.render("wordcloud_shape.html")
```

在上述示例中，我们首先导入了`WordCloud`类和相关的配置项模块`opts`。

然后，我们创建了一个`WordCloud`对象，并设置了图表的标题等。

接下来，我们添加了词云图的数据，数据以列表形式表示，每个元素是一个二元组，包含词语和对应的权重。

最后，我们使用`render`函数将图表渲染到一个HTML文件中。可以打开生成的`wordcloud_chart.html`文件来查看词云图的效果。

除了上述示例中的参数，`WordCloud()`函数还有其他可用的参数和配置选项，用于自定义词云图的样式、颜色、字体等。您可以根据您的需求进行进一步的定制和配置。

```python
def add(
    # 系列名称，用于 tooltip 的显示，legend 的图例筛选。
    series_name: str,

    # 系列数据项，[(word1, count1), (word2, count2)]
    data_pair: Sequence,

    # 词云图轮廓，有 'circle', 'cardioid', 'diamond', 'triangle-forward', 'triangle', 'pentagon', 'star' 可选
    shape: str = "circle",

    # 自定义的图片（目前支持 jpg, jpeg, png, ico 的格式，其他的图片格式待测试）
    # 该参数支持：
    # 1、 base64 （需要补充 data 头）；
    # 2、本地文件路径（相对或者绝对路径都可以）
    # 注：如果使用了 mask_image 之后第一次渲染会出现空白的情况，再刷新一次就可以了（Echarts 的问题）
    # Echarts Issue: https://github.com/ecomfe/echarts-wordcloud/issues/74
    mask_image: types.Optional[str] = None,

    # 单词间隔
    word_gap: Numeric = 20,

    # 单词字体大小范围
    word_size_range=None,

    # 旋转单词角度
    rotate_step: Numeric = 45,

    # 距离左侧的距离
    pos_left: types.Optional[str] = None,

    # 距离顶部的距离
    pos_top: types.Optional[str] = None,

    # 距离右侧的距离
    pos_right: types.Optional[str] = None,

    # 距离底部的距离
    pos_bottom: types.Optional[str] = None,

    # 词云图的宽度
    width: types.Optional[str] = None,

    # 词云图的高度
    height: types.Optional[str] = None,

    # 允许词云图的数据展示在画布范围之外
    is_draw_out_of_bound: bool = False,

    # 提示框组件配置项，参考 `series_options.TooltipOpts`
    tooltip_opts: Union[opts.TooltipOpts, dict, None] = None,

    # 词云图文字的配置
    textstyle_opts: types.TextStyle = None,

    # 词云图文字阴影的范围
    emphasis_shadow_blur: types.Optional[types.Numeric] = None,

    # 词云图文字阴影的颜色
    emphasis_shadow_color: types.Optional[str] = None,
)
```