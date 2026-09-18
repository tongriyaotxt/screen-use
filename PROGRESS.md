# PROGRESS — screen-use / desktop-jev

> 最后更新：2026-09-19 01:00（owner 睡前存档）。下次会话：先读本文，向 owner 汇报进展并继续。

## 一句话状态

**Jev SFT v0 已训完**：0.6B 决策模型在 5070Ti 上 27 分钟训完，留出集 op_acc 94.5% / joint_acc 68.8%，
τ=0.5 快路径覆盖率 70.9%；真实桌面冒烟时**主动低置信升级不乱点**（安全降级按设计工作）。
mock→real 泛化是下一个主战场。

## 已完成（倒序）

| 日期 | 内容 | 位置 |
|---|---|---|
| 09-19 | SFT v0（Qwen3-0.6B，3100 步，3 epochs）+ 校准 + serve.py | `checkpoints/jev-Qwen3-0.6B-e3/`（含 calibration.json） |
| 09-19 | mock v2 合成器：零干扰 + 15 类交互范式（54/54 oracle） | `scripts/mock_env.py`（--episodes/--seed/--start 断点续跑） |
| 09-18 | cascade protocol 阶段 A：`get_capabilities` + EscalationResult 带地图退 | `screen_use/capabilities.py`、`tools.py::run_task` |
| 09-18 | 阶段 0 轨迹采集（TraceRecorder）+ 阶段 1 首测（覆盖率 73.8-90.5% 成立） | `memory.py`、`scripts/analyze_traces.py` |
| 09-18 | 研究文档：desktop-jev 五件 + cascade-protocol | `research/desktop-jev/`、`research/cascade-protocol/` |
| 09-18 | 线上问题修复×4：焦点抢占(prompt规则5)/窗口激活验证/fail-safe复位/UIA配额 | `agent.py`、`tools.py`、`executor.py`、`uia_tree.py` |

测试：**118 passed**（`./.venv/Scripts/python.exe -m pytest -q`）

## 关键资产位置

- 模型：`checkpoints/jev-Qwen3-0.6B-e3/`（backbone + heads.pt + calibration.json；e1/e2 也在）
- 数据：`~/.screen_use/traces_mock/`（~3100 步 gold）、`~/.screen_use/traces_full/`（~105 步真实 silver）
- 推理：`training/serve.py` → `JevDecision(ckpt).decide(goal, foreground, elements, history)`
- 训练：`training/{common,dataset,model,train,calibrate}.py`

## 下一步（优先级 A > B > C，等 owner 指示或直接按 A 继续）

- **A 接入 run_task**：`serve.py` 接入 VisualLoop 快路径（协议位已在阶段 A 留好：
  `_think` 前插 Jev 决策，`escalate=True` 走原 VLM 路径）。先在 mock 环境实测
  "Jev 驾驶 vs VLM 驾驶"端到端速度对比（北极星指标首测）。
- **B 攒真实数据**：mock 量产只能在 **owner 离开电脑时**跑（弹层 grabMouse 教训）；
  日常用 run_task 自然攒 traces_full。
- **C SFT v1**：更多数据 / Qwen3-1.7B 基座，冲 joint_acc ≥80%；延迟 229ms→<100ms（量化）。

## 环境备忘（坑都踩过了）

- venv：`./.venv/Scripts/python.exe`（系统 python 无依赖）；PySide6/torch 都在 venv 里
- HF 下载：必须 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`（直连不通 + xet 401）
- GPU：RTX 5070 Ti Laptop（16GB），0.6B 全量微调 ~9.6GB
- mock 量产**绝不能在 owner 用电脑时跑**（弹层抓鼠标；窗口虽在角落但可见）
- 所有后台任务已停，无残留 MockUI 窗口

## 未 commit

本次会话改动未提交 git（等 owner 确认后提交）。
