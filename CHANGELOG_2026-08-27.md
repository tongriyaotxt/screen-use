# 变更记录 — 2026-08-27（VLM 调用降本 + 经验回放设计）

## 便宜改动（已实现，103 测试全绿 + 真机冒烟）

1. **反思不再重复截图**：`_reflect` 复用本轮 observe 的 SoM 标注截图（`agent.py`），
   省一次截图+编码，且标注图更利于 VLM 分析。
2. **`_think` 纠正式重试**：解析失败重试 2→1 次，但把解析错误原因反馈进 prompt
   （"你上次的输出无法解析…请只输出 JSON"），一次成功率更高。
3. **VLM 专用小图**：新增 `vlm_max_size` 配置（默认 896，env `VLM_MAX_SIZE`）。
   发给 VLM 的图不再用 1280——VLM 内部按 token 网格切图，小图显著降 TTFT；
   click_id 按元素 id 定位，不依赖分辨率。作用于 run_task / find_element / read_screen。
4. **UIA 枚举与截图并行**：`screenshot(annotate=True)` 中 UIA 枚举放线程
   （COM 需 CoInitialize），与截图/缩放并行；线程失败自动回退主线程顺序枚举。
   真机冒烟：0.38s / 64 元素。

## 经验回放设计（未实现，研究文档）

- 新增 `research/experience-replay/`：`DESIGN.md`（设计方案）、
  `literature.md`（文献调研，含 AppAgentX/Voyager/RTC 等量化证据）、
  `gaps.md`（空白分析：三件套组合无先例、具身迁移的学术风险）。
- 核心主张：轨迹 → 语义锚点参数化宏 → 检查点验证 → bailout 回退 VLM 循环；
  重复任务 VLM 调用从 N 次降到 ~3 次。README roadmap 已挂链接。

## 测试

103 passed（97 + 新增 6：纠错重试、反思复用截图、vlm_max_size、并行枚举及兜底）。
