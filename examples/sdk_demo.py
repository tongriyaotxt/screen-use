"""SDK 演示：直接用 Python API 操作桌面。

运行前：
1. 复制 .env.example 为 .env（不配 VLM 也能跑 UIA 匹配路径）
2. 手动打开一个计算器（Win+R 输入 calc）
3. python examples/sdk_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screen_use.tools import ScreenUse

tools = ScreenUse()  # dry_run=True 可先观察动作记录不真实点击

print("== 前台窗口的可交互元素 ==")
for el in tools.list_ui_elements()[:10]:
    print(f"  #{el['id']} [{el['control_type']}] {el['name']!r} center={el['center']}")

print("\n== 策略链定位：找数字 2 按钮 ==")
result = tools.find_element("2")
print(f"  found={result['found']} method={result.get('method')}")

if result["found"]:
    print("\n== 点击 ==")
    click_result = tools.click_element("2")
    print(f"  ok={click_result['action']['ok']}")
