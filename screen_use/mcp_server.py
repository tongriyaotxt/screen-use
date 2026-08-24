"""MCP Server：把 ScreenUse 暴露为 MCP 工具，任何支持 MCP 的 Agent 即插即用。

启动方式：
    python -m screen_use.mcp_server          # stdio 模式（Claude Desktop / Kimi CLI 等）
    rpa-mcp-server                    # 安装后的命令入口
"""

from __future__ import annotations

import base64
import json

# mcp 2.x：FastMCP 更名为 MCPServer
from mcp.server.mcpserver import Image, MCPServer

from screen_use.tools import ScreenUse

mcp = MCPServer("screen-use")
tools = ScreenUse()


# ================= 原子工具 =================

@mcp.tool()
def screenshot(annotate: bool = False):
    """截取当前屏幕。annotate=True 时在图上叠加编号框并返回可交互元素列表。"""
    result = tools.screenshot(annotate=annotate)
    img_bytes = base64.b64decode(result.pop("image_base64"))
    return [Image(data=img_bytes, format="jpeg"), json.dumps(result, ensure_ascii=False)]


@mcp.tool()
def list_ui_elements() -> str:
    """枚举前台窗口的可交互控件（id、名称、类型、坐标）。零模型依赖，速度快。"""
    return json.dumps(tools.list_ui_elements(), ensure_ascii=False)


@mcp.tool()
def click(x: int, y: int) -> str:
    """点击屏幕坐标 (x, y)。"""
    return json.dumps(tools.click(x, y), ensure_ascii=False)


@mcp.tool()
def double_click(x: int, y: int) -> str:
    """双击屏幕坐标 (x, y)。"""
    return json.dumps(tools.double_click(x, y), ensure_ascii=False)


@mcp.tool()
def right_click(x: int, y: int) -> str:
    """右键点击屏幕坐标 (x, y)。"""
    return json.dumps(tools.right_click(x, y), ensure_ascii=False)


@mcp.tool()
def click_element_id(element_id: int) -> str:
    """点击 list_ui_elements 返回的元素 id（使用最近一次枚举的缓存）。"""
    return json.dumps(tools.click_element_id(element_id), ensure_ascii=False)


@mcp.tool()
def type_text(text: str) -> str:
    """在当前焦点处输入文本（支持中文，自动走剪贴板粘贴）。"""
    return json.dumps(tools.type_text(text), ensure_ascii=False)


@mcp.tool()
def hotkey(keys: str) -> str:
    """按组合键，keys 用 + 分隔，如 "ctrl+s"、"alt+tab"。"""
    return json.dumps(tools.hotkey(*keys.split("+")), ensure_ascii=False)


@mcp.tool()
def press(key: str) -> str:
    """按单个键，如 "enter"、"esc"、"tab"。"""
    return json.dumps(tools.press(key), ensure_ascii=False)


@mcp.tool()
def scroll(clicks: int, x: int | None = None, y: int | None = None) -> str:
    """滚动鼠标滚轮。clicks 正数向上、负数向下；可选指定滚动位置。"""
    return json.dumps(tools.scroll(clicks, x, y), ensure_ascii=False)


# ================= 高层工具 =================

@mcp.tool()
def find_element(description: str) -> str:
    """按自然语言描述定位 UI 元素（如 "保存按钮"）。

    内部策略链：UIA 文本匹配（零模型依赖）→ SoM + VLM 视觉定位。
    返回元素的坐标、名称和命中方式。
    """
    return json.dumps(tools.find_element(description), ensure_ascii=False)


@mcp.tool()
def click_element(description: str) -> str:
    """按自然语言描述定位并点击 UI 元素，一步完成（如 "点击确定按钮"）。"""
    return json.dumps(tools.click_element(description), ensure_ascii=False)


@mcp.tool()
def read_screen(question: str) -> str:
    """就当前屏幕内容向 VLM 提问（如 "这个弹窗说了什么？"）。需要配置 VLM。"""
    return tools.read_screen(question)


@mcp.tool()
def run_task(goal: str, max_steps: int = 20) -> str:
    """视觉行为循环：自主完成一个桌面任务（如 "打开计算器算 25*4"）。

    循环执行 观察截图→VLM决策→动作→验证，直到完成或达到步数上限。需要配置 VLM。
    """
    return json.dumps(tools.run_task(goal, max_steps=max_steps), ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
