# PROGRESS — screen-use / desktop-jev

> 最后更新：2026-09-19 10:50（阶段 4 接入 + 北极星首测完成）。下次会话：先读本文，向 owner 汇报进展并继续。

## 一句话状态

**Jev 已接进 VisualLoop 快路径并跑通首测**：同一批 mock 任务每任务 **17.7s → 10.2s（1.74×）**，
Jev 决策 p50 **43ms** vs VLM **5978ms**（139×）。但实测暴露两个新瓶颈：
**观测（UIA 枚举 0.68s/步）**与**升级率（32–36%，压 TRAINING §7 的 35% 及格线）**。
另修掉一个会让 τ 语义失效的校准 bug（详见 `research/desktop-jev/BENCH.md` §4）。

## 已完成（倒序）

| 日期 | 内容 | 位置 |
|---|---|---|
| 09-19 | **阶段 4 接入 + 北极星首测**（1.74×，报告见 BENCH.md） | `screen_use/jev_bridge.py`、`agent.py`、`scripts/bench_jev.py`、`scripts/eval_offline.py`、`scripts/smoke_jev_loop.py`、`scripts/bench_gui.py` |
| 09-19 | 修校准接入 bug：等张断点落盘 + serve 侧插值；留出集再分半（半拟合/半选 τ）；重校准 τ=0.77 | `training/{calibrate,serve}.py`、`checkpoints/jev-Qwen3-0.6B-e3/calibration.json` |
| 09-19 | serve 补 TYPE 的 goal-copy 文本（span 头解码） | `training/serve.py` |
| 09-19 | SFT v0（Qwen3-0.6B，3100 步，3 epochs）+ 校准 + serve.py | `checkpoints/jev-Qwen3-0.6B-e3/` |
| 09-19 | mock v2 合成器：零干扰 + 15 类交互范式（54/54 oracle） | `scripts/mock_env.py`（--episodes/--seed/--start 断点续跑） |
| 09-18 | cascade protocol 阶段 A：`get_capabilities` + EscalationResult 带地图退 | `screen_use/capabilities.py`、`tools.py::run_task` |
| 09-18 | 阶段 0 轨迹采集（TraceRecorder）+ 阶段 1 首测（覆盖率 73.8-90.5% 成立） | `memory.py`、`scripts/analyze_traces.py` |
| 09-18 | 研究文档：desktop-jev 五件 + cascade-protocol | `research/desktop-jev/`、`research/cascade-protocol/` |
| 09-18 | 线上问题修复×4：焦点抢占/窗口激活验证/fail-safe复位/UIA配额 | `agent.py`、`tools.py`、`executor.py`、`uia_tree.py` |

测试：**155 passed**（`./.venv/Scripts/python.exe -m pytest tests/ -q`，118 → 155，新增 Jev 快路径/升级/认输/异常降级/词表同源等护栏）

## 关键指标（09-19 首测，mock；原始数据 `assets/bench/`）

| 指标 | 值 | 备注 |
|---|---|---|
| 端到端（每任务） | 纯 VLM 17.7s → **Jev 混合 10.2s（1.74×）** | 4 个回合配对，观测/执行用本机实测常量 |
| Jev 决策时延 | p50 **43ms** / mean 47ms（另一轮 300 样本单跑 p50 116ms，GPU 时钟与元素表规模不同） | 目标 <100ms 基本达标 |
| VLM 决策时延 | p50 **5978ms** / mean 6200ms（n=10，真 API） | 对照组 |
| 离线 op_acc | **0.962 ± 0.018**（3 个 10% 切片） | 训练日志 0.945 同量级 |
| 离线 joint_acc | **0.758 ± 0.061** | 切片方差大；单切片（如 0.688）不可当结论 |
| 部署侧覆盖率 @τ=0.77 | **0.677**（n=300 等距抽样）· 升级率 **0.323** | 校准文件里的 0.759 是 10% 切片，偏乐观 |
| 推荐 τ | **0.72**（覆盖率 0.743） | 零训练成本，改 `calibration.json` 即可 |
| mock 数据 | 1125 文件 / 3100 步 gold | 前 656 个是 v1 合成器（6 类），后 469 是 v2（15 类） |
| 真实数据 | 9 文件 / 105 步 silver | 仍太少；其中 hotkey/press 步不进训练集 |

## 关键资产位置

- 模型：`checkpoints/jev-Qwen3-0.6B-e3/`（backbone + heads.pt + calibration.json 含 isotonic 断点）
- 推理：`training/serve.py` → `JevDecision(ckpt).decide(goal, foreground, elements, history)`
  → `{action, element_id, text, confidence, confidence_raw, escalate, latency_ms}`
- 桥接：`screen_use/jev_bridge.py`（OPS 单一真源 / 动作翻译 / history 同构 / 引擎懒加载）
- 接入开关：`.env` 里 `JEV_CHECKPOINT=checkpoints/jev-Qwen3-0.6B-e3`（**默认留空 = 纯 VLM**）
- 基准：`scripts/bench_jev.py`（无头 replay）· `scripts/eval_offline.py`（多切片+部署曲线）·
  `scripts/smoke_jev_loop.py`（真桌面 dry-run 冒烟，零鼠标零外发）
- 报告：`research/desktop-jev/BENCH.md` · 原始 JSON/MD：`assets/bench/`

## 下一步（优先级 A > B > C，等 owner 指示或直接按 A 继续）

- **A 压升级率（零训练成本）**：✅ 已完成 2026-09-19 —— `training.calibrate --tau 0.72`
  钉死（`calibration.json` 标 `tau_source: cli`），部署复核：覆盖率 0.743 / 升级率 0.257 /
  风险 0.103 / 单步 p50 38.9ms。
- **B 砍观测成本**：✅ 主体完成 2026-09-19。体检 + 四条路实测见 `research/desktop-jev/BENCH.md` §6：
  后台窗口**元素表缓存**（默认开，TTL 2s，冷启 326ms → 热 153ms，元素表不缩水）·
  升级步复用元素表（默认开）· 时间预算开关 `SCREEN_USE_UIA_BUDGET_MS`（默认 0=不限）·
  多窗口并行无收益已回退。点击路径的安全阀：点/写前实时 bbox 校验，移位则刷新重解析，
  再找不到明确报错（`element_still_valid` / `_fresh_element`）。
- **C 真 GUI 端到端**：**只能在 owner 离开电脑时跑**。`scripts/bench_gui.py` 已备好
  （mock 窗口 + 零鼠标执行器 + 真 VisualLoop；`--driver oracle|jev|vlm`；默认拒跑，需 `--allow-gui`；
  `--offscreen` 自检已过）。首次真窗口运行建议人在旁边。
- **D SFT v1**：joint_acc 0.758 → ≥0.80；1.7B 基座 / 更多真实数据（腿 1 仍只有 105 步）。
- **E 攒真实数据**：日常用 run_task 自然攒 `traces_full`（现在快路径步会标 `source=jev`）。

## 环境备忘（坑都踩过了）

- venv：`./.venv/Scripts/python.exe`（系统 python 无依赖）；PySide6/torch 都在 venv 里
- HF 下载：必须 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`（直连不通 + xet 401）
- GPU：RTX 5070 Ti Laptop（16GB）。**两个 torch 进程同时跑会互相拖慢**（实测 Jev 时延翻倍），
  基准测量必须串行；`scripts/eval_offline.py` 与 `bench_jev.py` 别并发
- mock 量产**绝不能在 owner 用电脑时跑**（弹层抓鼠标；窗口虽在角落但可见）
- 本机 **PATH 里没有 git**（`git` 命令不存在）→ 提交需要 owner 指路 git.exe
- PowerShell 传多行脚本给 `python -c` 会丢引号：写成临时 .py 文件再跑
- 所有后台任务已停，无残留 MockUI 窗口

## 未 commit / git 备忘

- 已提交到本地 `main`（`git log` 看提交信息）；**owner 定的 push 闸门：做到成品
  （常用 GUI 通用可用 + 端到端快于人类基线）并经 owner 确认才 push** —— 当前 ahead of origin。
- `git` 不在 PATH，二进制在 `C:\Program Files\Git\cmd\git.exe`。
- `.gitignore` 忽略 `checkpoints/`（权重太大）→ **`calibration.json` 的 τ=0.72 不在版本控制里**，
  换机/重训后要重新 `training.calibrate --tau 0.72`（参数已支持）。
- `*.log` 也被忽略，基准日志在 `assets/bench/logs/`（本地留存，不入库）。
