"""三环能力说明书（cascade protocol §3）。

宿主 agent 启动时调用 get_capabilities()，据此把任务拆成各环能干的形状：
- skill：已验证宏的确定性回放（experience-replay，未实现 → available=False）
- task：步级决策循环（终态 Jev 小模型；当前引擎 = VLM 视觉循环）
- atomic：原子鼠标键盘工具，宿主完全控制

诚实原则：can/cannot 描述的是**契约**，status/engine 标注**当前实现**；
Jev 训练完成后按实测数据回填（research/cascade-protocol/README.md §7）。
"""

from __future__ import annotations

CAPABILITIES: dict = {
    "version": "0.1.0",
    "protocol": "cascade-protocol/0.1",
    "rings": {
        "skill": {
            "description": "已验证宏的确定性回放，零模型，最快。仅覆盖已录制任务。",
            "available": False,
            "status": "planned（experience-replay，未实现）",
            "tools": ["list_skills", "run_skill"],
        },
        "task": {
            "description": "自主任务循环：给一个目标，内部逐步决策直到完成或升级。",
            "available": True,
            "engine": "vlm（每步 3-30s；Jev 小模型接入后 ~0.1s/步，协议不变）",
            "tools": ["run_task"],
            "can": [
                "点击/滚动/等待/判断完成（click/scroll/wait/done）",
                "单窗口或明确跨窗口的元素定位与导航",
                "表单填写：值必须显式出现在 goal 中（copy，不创作）",
            ],
            "cannot": [
                "生成自由文本（总结、回复、写作）——请宿主自己生成文本后放入 goal",
                "视觉内容理解（读图表、识别验证码）",
                "处理意外弹窗/未知对话框",
                "不可逆操作（发送/删除/付款）——宿主应先用原子工具自行确认",
                "多步推理链（'比较 A 和 B 再决定'）——请宿主拆成原子子任务",
            ],
            "goal_style": (
                "单句祈使句，参数显式给出。"
                "例：'在记事本中输入：TODO 周五交报告'，而非'帮我写个备忘录'"
            ),
        },
        "atomic": {
            "description": "原子鼠标键盘工具（18 个），宿主完全控制。任何任务可用，速度取决于宿主。",
            "available": True,
            "tools": [
                "screenshot", "list_ui_elements", "click", "click_scaled",
                "double_click", "right_click", "click_element_id",
                "get_element_text", "set_element_text", "type_text",
                "hotkey", "press", "scroll", "do_actions",
                "find_element", "click_element", "read_screen",
            ],
        },
    },
    "escalation": (
        "run_task 无法完成时不抛错，返回 EscalationResult："
        "{status: 'escalated', reason: {class, detail}, progress, observation, suggestion}。"
        "observation 含当前元素表，宿主可直接决策下一步，无需重新截图枚举。"
    ),
    "routing_hint": (
        "1. 有宏用宏（list_skills 先查）；2. 任务符合 task.can 且无 task.cannot → run_task；"
        "3. 需要创作文本/视觉理解/不可逆操作 → 宿主用原子工具直接驾驶。"
    ),
}


def get_capabilities() -> dict:
    """返回三环能力说明书（见模块 docstring）。"""
    return CAPABILITIES
