"""端到端演示：自动操作计算器计算 2 + 2 并读取结果。

验证 ScreenUse 的完整链路：UIA 枚举 → 策略链定位 → 真实点击 → 结果校验。
全程零模型调用（UIA 文本匹配命中），无需配置 VLM。

运行：python examples/demo_calculator.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uiautomation as auto

from screen_use.tools import ScreenUse

auto.SetGlobalSearchTimeout(5)


def main() -> None:
    # 1. 启动并激活计算器
    os.startfile("calc.exe")
    time.sleep(2)
    win = auto.WindowControl(searchDepth=1, SubName="Calculator")
    if not win.Exists(3):
        win = auto.WindowControl(searchDepth=1, SubName="计算器")
    win.SetActive()
    time.sleep(0.5)

    tools = ScreenUse()

    # 2. 用自然语言描述依次点击：Two → Plus → Two → Equals
    for step in ["Two", "Plus", "Two", "Equals"]:
        result = tools.click_element(step)
        assert result["found"], f"未找到按钮: {step}"
        print(
            f"点击 {step!r}: method={result['method']} "
            f"name={result['element']['name']!r} center={result['element']['center']}"
        )
        time.sleep(0.3)

    # 3. 读取计算结果（显示区是只读 TextControl，直接从 UIA 树取）
    time.sleep(0.5)
    display = win.TextControl(AutomationId="CalculatorResults")
    result_text = display.Name
    print(f"\n计算器显示: {result_text!r}")

    # 4. 断言结果包含 4（中英文系统分别显示 "显示为 4" / "Display is 4"）
    assert "4" in result_text, f"结果不正确: {result_text}"
    print("✅ 验证通过：2 + 2 = 4")


if __name__ == "__main__":
    main()
