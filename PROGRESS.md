# PROGRESS — screen-use / desktop-jev

> 最后更新：2026-09-19 13:45（v3 流水线已挂后台跑，等 owner 离开电脑自动推进）。
> 下次会话：先读本文 + `assets/bench/logs/pipeline_*.log`，向 owner 汇报并继续。

## 一句话状态

**Jev 已接进 VisualLoop 快路径并跑通首测**：mock 域每任务 **17.7s → 10.2s（1.74×）**，
Jev 决策 p50 **40ms** vs VLM **5978ms**。
但真实域首测（traces_full 76 样本）暴露硬伤：**真实桌面元素表中位 84（mock 14）、token 2380（mock 457），
Jev 的 joint_acc 从 0.77 掉到 0.17、升级率 90%** —— 泛化是当前第一瓶颈。
已加**训练分布门控**（观测超出训练尺度就不问模型，直接升级）：真实域 Jev 侧耗时 452ms → **0ms**，
mock 域覆盖率不变（0.763）。诚实结论：**今天在真实应用上快路径的价值是 0、安全是满分**；
要达成 owner 的成品标准（通用可用 + 快于人类），下一步必须补数据分布。

## 已完成（倒序）

| 日期 | 内容 | 位置 |
|---|---|---|
| 09-19 | **真画图标**（owner 报「我没看到纯图标」）：v3 所谓"图标按钮"其实是 `QPushButton("")` **空白方块**；真实那批是窗口按钮 ─ □ ✕（实测 945/9284=10.2% 无名、全是 ButtonControl、bbox 69×60 贴右上角）。现 `_glyph_icon()` 画通用几何图形（**刻意不画语义图标**，免得诱导 VLM 点错污染对照）+ 「无名比例归零」告警护栏 | `scripts/mock_env.py`、`scripts/mock_v3.py`、`tests/test_mock_windows.py`、BENCH.md §8.7 |
| 09-19 | **修训练集标签污染**：`iter_samples` 的 docstring 写着「跳过…执行失败步」，代码里**没有这个判断** → 失败步（`args` 为空）被当成「CLICK 无目标元素」的**正样本**。实测 v3 14/290、v2 68/3100、真实 25/105 都中招（**v0 那轮就吃进去了**）。修后 v3 样本 290→276 | `training/common.py`、`tests/test_training_common.py`（+5 测试） |
| 09-19 | **修窗口卡死事故**（owner 报障「哪个 mock 卡住了显示未响应」）：确认框是 QDialog 且**故意与主窗口同标题**，有些范式流程结束时把它留在打开状态，而 `win.close()` 只隐藏主窗口、关不掉独立顶层窗口 → 那个框赖在 owner 屏幕上，生成器进 idle-gate 阻塞事件循环后被 Windows 标成「(未响应)」。现 `MockWindow.teardown()` 按层收（弹层→确认框→本体）+ 回合边界 `sweep_stale_windows()` 兜底；进闸门前必清场 | `scripts/mock_env.py`、`scripts/mock_v3.py`、`tests/test_mock_windows.py`（+4 测试） |
| 09-19 | **v3 流水线开跑**（1000 回合 / 2 epoch，idle-gate 60s，你离开电脑才动） | `scripts/run_pipeline.py`、`assets/bench/logs/pipeline_*.log` |
| 09-19 | **修流水线评测域**：第 5 步 mock 基准漏传 `--data` → 会拿 v2（14 元素）考 v3（90 元素）模型，低分会被误读成"训练搞坏了" | `scripts/run_pipeline.py` |
| 09-19 | **修校准小样本崩溃**：哈希分半可能把留出文件全分到一侧 → `torch.cat([])` 报错看不懂；现按文件整体挪一个 + 明确错误信息 | `training/calibrate.py`、`tests/test_calibrate.py`（+6 测试） |
| 09-19 | **图标无文字分布对齐**：真实 11% 元素无名 → v3 掺无名图标按钮 + 无重名保护 | `scripts/mock_v3.py`、`tests/test_mock_v3.py`、BENCH.md §8.7/8.8 |
| 09-19 | **人类基线工具**（8 个可验证日常任务，人/agent 同口径对照） | `scripts/baseline.py`、`tests/test_baseline.py` |
| 09-19 | **mock v3 生成器 + 一条命令流水线**（尺度/语义/双语对齐真实分布） | `scripts/mock_v3.py`、`scripts/run_pipeline.py`、`tests/test_mock_v3.py` |
| 09-19 | **真实域泛化首测（R4）**：joint_acc 0.17 / 升级率 90% → 加训练分布门控 | `screen_use/jev_bridge.py::ood_check`、`training/calibrate.py::train_stats`、`assets/bench/offline_eval_realdomain.json` |
| 09-19 | **后台窗口元素表缓存 + 点击前 bbox 校验**（观测 326ms → 153ms） | `screen_use/perception/uia_tree.py`、`tools.py::_fresh_element` |
| 09-19 | **阶段 4 接入 + 北极星首测**（1.74×，报告见 BENCH.md） | `screen_use/jev_bridge.py`、`agent.py`、`scripts/bench_jev.py`、`scripts/eval_offline.py`、`scripts/smoke_jev_loop.py`、`scripts/bench_gui.py` |
| 09-19 | 修校准接入 bug：等张断点落盘 + serve 侧插值；留出集再分半（半拟合/半选 τ）；重校准 τ=0.77 | `training/{calibrate,serve}.py`、`checkpoints/jev-Qwen3-0.6B-e3/calibration.json` |
| 09-19 | serve 补 TYPE 的 goal-copy 文本（span 头解码） | `training/serve.py` |
| 09-19 | SFT v0（Qwen3-0.6B，3100 步，3 epochs）+ 校准 + serve.py | `checkpoints/jev-Qwen3-0.6B-e3/` |
| 09-19 | mock v2 合成器：零干扰 + 15 类交互范式（54/54 oracle） | `scripts/mock_env.py`（--episodes/--seed/--start 断点续跑） |
| 09-18 | cascade protocol 阶段 A：`get_capabilities` + EscalationResult 带地图退 | `screen_use/capabilities.py`、`tools.py::run_task` |
| 09-18 | 阶段 0 轨迹采集（TraceRecorder）+ 阶段 1 首测（覆盖率 73.8-90.5% 成立） | `memory.py`、`scripts/analyze_traces.py` |
| 09-18 | 研究文档：desktop-jev 五件 + cascade-protocol | `research/desktop-jev/`、`research/cascade-protocol/` |
| 09-18 | 线上问题修复×4：焦点抢占/窗口激活验证/fail-safe复位/UIA配额 | `agent.py`、`tools.py`、`executor.py`、`uia_tree.py` |

测试：**230 passed**（`./.venv/Scripts/python.exe -m pytest tests/ -q`，118 → 155 → 230，新增 Jev 快路径/升级/认输/异常降级/词表同源/校准分半等护栏）

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
| 真实数据 | 9 文件 / 105 步 silver | 仍太少；剔除失败步与动作集外的步后**只有 72 条可训**（见 09-19 标签污染修复） |

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

- **A 补数据分布（第一优先级，理由见 BENCH.md §7）**：生成器已写好（`scripts/mock_v3.py`，
  元素中位 87 对齐真实 84 / goal 语义化不照抄控件名 / 中英混排命名），
  **状态：2026-09-19 13:40 已挂后台跑**（`--episodes 1000 --epochs 2 --idle-gate-s 60
  --throttle-ms 50`，owner 开着电脑就静默停着、不弹窗；日志 `assets/bench/logs/pipeline_20260919-1340.log`）。
  **跑法（owner 离开电脑时，一条命令）**：
  ```
  ./.venv/Scripts/python.exe scripts/run_pipeline.py --i-am-away            # 3000 回合 + 3 epoch
  ./.venv/Scripts/python.exe scripts/run_pipeline.py --i-am-away --episodes 200 --epochs 1  # 小规模验管线
  ```
  自动串起：v3 量产（断点续跑）→ 训练（输出到 `checkpoints/v3`，**不覆盖 v0**）→ 校准 →
  真实域复测 → mock 基准 → 写对照表 `assets/bench/pipeline_<ts>.md`。
  验收看**真实域** joint_acc / 升级率有没有动（mock 涨了不算数）。
  再做：学出来的两阶段决策（小打分器把 80+ 元素缩到 ~20 候选；
  **字符串短名单已证伪**：真实域 recall@10=0%，见 BENCH.md §7）。
- **B 真 GUI 端到端**：**只能在 owner 离开电脑时跑**。`scripts/bench_gui.py` 已备好
  （mock 窗口 + 零鼠标执行器 + 真 VisualLoop；`--driver oracle|jev|vlm`；默认拒跑，需 `--allow-gui`；
  `--offscreen` 自检已过）。首次真窗口运行建议人在旁边。
- **C 通用性 + 人类基线**（成品标准的两个验收项）：
  - 人类基线工具已就绪：`scripts/baseline.py --mode human`（交互计时，几分钟）→
    `--mode agent --i-am-away`（同 goal 同验证器真跑）→ `--mode report`（中位倍数 = 北极星单一数字）。
    8 个任务覆盖记事本/计算器/资源管理器/浏览器，文件类程序化验证。
  - 通用性实测：按应用族（浏览器/Office/聊天/文件管理器/IDE）各跑一组可验证任务，
    记录成功率与升级率。
- **D 攒真实数据**：日常用 run_task 自然攒 `traces_full`（快路径步标 `source=jev`）。

## 环境备忘（坑都踩过了）

- venv：`./.venv/Scripts/python.exe`（系统 python 无依赖）；PySide6/torch 都在 venv 里
- HF 下载：必须 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`（直连不通 + xet 401）
- GPU：RTX 5070 Ti Laptop（16GB）。**两个 torch 进程同时跑会互相拖慢**（实测 Jev 时延翻倍），
  基准测量必须串行；`scripts/eval_offline.py` 与 `bench_jev.py` 别并发
- mock 量产**绝不能在 owner 用电脑时跑**（弹层抓鼠标；窗口虽在角落但可见）
- 本机 **PATH 里没有 git**（`git` 命令不存在）→ 用 `C:\Program Files\Git\cmd\git.exe`
- **别用 PowerShell 的 `Get-Content | Set-Content` 改 UTF-8 源码**：PS5.1 按 ANSI(GBK) 读，
  中文会变乱码且部分字节不可逆（本次毁过一次测试文件，只能整份重写）。改文件一律用编辑工具，
  或 python 显式 `encoding="utf-8"` 读写。
- PowerShell 传多行脚本给 `python -c` 会丢引号：写成临时 .py 文件再跑
- 两个 torch 进程同时跑会互相拖慢（基准测量必须串行）

## 未 commit / git 备忘

- 已提交到本地 `main`（`git log` 看提交信息）；**owner 定的 push 闸门：做到成品
  （常用 GUI 通用可用 + 端到端快于人类基线）并经 owner 确认才 push** —— 当前 ahead of origin。
- `git` 不在 PATH，二进制在 `C:\Program Files\Git\cmd\git.exe`。
- `.gitignore` 忽略 `checkpoints/`（权重太大）→ **`calibration.json` 的 τ=0.72 不在版本控制里**，
  换机/重训后要重新 `training.calibrate --tau 0.72`（参数已支持）。
- `*.log` 也被忽略，基准日志在 `assets/bench/logs/`（本地留存，不入库）。
