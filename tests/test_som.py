"""SoM 标注测试：合成图片，不依赖真实桌面。"""

from PIL import Image

from screen_use.perception import som
from screen_use.perception.uia_tree import UIElement


def _make_elements():
    return [
        UIElement(id=1, name="确定", control_type="ButtonControl", bbox=(10, 10, 60, 40)),
        UIElement(id=2, name="取消", control_type="ButtonControl", bbox=(70, 10, 120, 40)),
    ]


def test_annotate_returns_same_size_image():
    img = Image.new("RGB", (200, 100), "white")
    out = som.annotate(img, _make_elements())
    assert out.size == img.size
    # 原图不被修改
    assert img.tobytes() != out.tobytes()


def test_annotate_draws_boxes():
    img = Image.new("RGB", (200, 100), "white")
    out = som.annotate(img, _make_elements())
    # 框的边缘像素应被染色（非白色）
    edge_pixel = out.getpixel((10, 25))  # 元素1 左边框中点
    assert edge_pixel != (255, 255, 255)


def test_annotate_with_scale():
    img = Image.new("RGB", (100, 50), "white")
    elements = _make_elements()  # 物理像素 bbox
    out = som.annotate(img, elements, scale=2.0)  # 图是物理像素的一半
    # bbox (10,10) / 2 = (5,5)，该处应被染色
    assert out.getpixel((5, 15)) != (255, 255, 255)


def test_legend_format():
    text = som.legend(_make_elements())
    assert "1: [ButtonControl] 确定" in text
    assert "2: [ButtonControl] 取消" in text


def test_legend_handles_unnamed():
    elements = [UIElement(id=1, name="", control_type="EditControl", bbox=(0, 0, 10, 10))]
    assert "(无名)" in som.legend(elements)
