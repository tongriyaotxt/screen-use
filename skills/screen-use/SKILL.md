---
name: screen-use
description: 操作 Windows 桌面应用（点击、输入、截图、自动化任务）。当用户要求操作桌面软件、点击按钮、填写表单、跨应用搬运数据、或完成任何 GUI 自动化任务时使用。工具来自 screen-use MCP server。
---

# screen-use — 桌面自动化

通过 `screen-use` MCP 工具操作 Windows 桌面。你能看屏幕、找控件、点击输入。

## 工具选择策略（重要）

按优先级选择，能少走很多弯路：

1. **`click_element("描述")` 是首选** —— 一句话定位并点击，内部走 UIA 文本匹配（毫秒级、零模型），如 `click_element("保存")`、`click_element("确定")`
2. **`list_ui_elements`** 查看当前所有可交互控件（含窗口名、坐标），需要了解界面结构时用
3. **`click_element_id(id)`** 点击上面列表里的编号（点击前会自动激活所属窗口）
4. **`click(x, y)` 裸坐标** 是最后手段 —— 先 `screenshot()` 看图，再估算坐标
5. **`screenshot(annotate=True)`** 带编号框标注的截图，适合你自己（多模态）看图分析

## 输入与快捷键

- `type_text("文本")` —— 中英文都支持（剪贴板粘贴），**输入前先点击目标输入区获得焦点**
- `hotkey("ctrl+s")`、`press("enter")` —— 快捷键和单键
- `scroll(-3)` 向下滚动

## 自主任务

- `run_task("任务描述")` —— VLM 视觉循环自主完成多步任务。**需要本地 Ollama 或配置云端 VLM（含 Kimi Code 订阅，`VISION_PROVIDER=kimi-code` 零额外成本）**，每步决策 10-30 秒，适合复杂且路径不确定的任务
- 简单任务（3 步以内、目标明确）优先自己用 `click_element` 组合完成，更快更稳

## 工作准则

- **每步后验证**：重要操作后 `screenshot()` 或 `list_ui_elements` 确认结果
- **切换窗口**：直接点击目标窗口的元素即可（自动激活），不要用 alt+tab
- **紧急停止**：用户把鼠标甩到屏幕左上角会触发 failsafe 急停
- 元素 id 会随界面刷新失效，找不到时重新 `list_ui_elements`

## 典型流程示例

"把记事本内容清空" →
1. `click_element("记事本编辑区")` 或用 `list_ui_elements` 找到 DocumentControl
2. `hotkey("ctrl+a")` + `press("delete")`
3. `screenshot()` 确认已清空
