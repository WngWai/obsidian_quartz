make_snapshot渲染为图片

```python
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot

# 使用 snapshot-selenium 渲染图片
from snapshot_selenium import snapshot

bar = (
    Bar()
    .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
)
make_snapshot(snapshot, bar.render(), "bar.png")
```


```python
def make_snapshot(
    # 渲染引擎，可选 selenium 或者 phantomjs
    engine: Any,

    # 传入 HTML 文件路径
    file_name: str,

    # 输出图片路径
    output_name: str,

    # 延迟时间，避免图还没渲染完成就生成了图片，造成图片不完整
    delay: float = 2,

    # 像素比例，用于调节图片质量
    pixel_ratio: int = 2,

    # 渲染完图片是否删除原 HTML 文件
    is_remove_html: bool = False,

    # 浏览器类型，目前仅支持 Chrome, Safari，使用 snapshot-selenium 时有效
    browser: str = "Chrome",
    **kwargs,
)
```



