"""Set-of-Mark 标注：在截图上为候选元素绘制编号框。

VLM 只需回答"点几号框"（选择题），而非预测像素坐标（填空题），
大幅降低对模型定位能力的要求。
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from screen_use.perception.uia_tree import UIElement

# 调色板：高对比度，避免与常见 UI 撞色
_PALETTE = [
    "#FF3B30", "#007AFF", "#34C759", "#FF9500", "#AF52DE",
    "#FF2D55", "#5AC8FA", "#FFCC00", "#5856D6", "#00C7BE",
]

# 标注图中的坐标都是缩放图坐标；elements 的 bbox 是物理像素，需除以 scale
def annotate(
    img: Image.Image,
    elements: list[UIElement],
    scale: float = 1.0,
) -> Image.Image:
    """在 img 副本上绘制编号框。scale = 物理像素 / 图上像素。"""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    for el in elements:
        left, top, right, bottom = [int(v / scale) for v in el.bbox]
        color = _PALETTE[(el.id - 1) % len(_PALETTE)]

        # 编号框
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        # 编号标签（白底彩字，放在框左上角）
        label = str(el.id)
        tw = draw.textlength(label, font=font)
        th = 12
        label_box = [left, max(0, top - th - 4), left + int(tw) + 6, max(th + 4, top)]
        draw.rectangle(label_box, fill=color)
        draw.text((label_box[0] + 3, label_box[1] + 1), label, fill="white", font=font)

    return out


def legend(elements: list[UIElement]) -> str:
    """生成给 VLM 的图例文本：编号 → 控件名/类型。"""
    lines = []
    for el in elements:
        name = el.name or "(无名)"
        lines.append(f"{el.id}: [{el.control_type}] {name}")
    return "\n".join(lines)
