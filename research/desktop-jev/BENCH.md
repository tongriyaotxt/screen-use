# 阶段 4：Jev 接入 VisualLoop + 北极星指标首测

> 2026-09-19 · 承接 `DESIGN.md` §10 阶段 4「校准 + 接入」、`TRAINING.md` §7 在线指标、
> `../cascade-protocol/README.md` 阶段 C 与 §4 升级协议。
> 原始数据：`assets/bench/jev_vs_vlm_*.{json,md}`、`assets/bench/offline_eval.json`。

## 0. 一句话结论

**接通了，且比现状快 1.7×（同一批 mock 任务每任务 17.7s → 10.2s）**；
但真正的瓶颈已经不是决策（Jev 43–116 ms/步 vs VLM 6.0 s/步），而是**观测**（UIA 枚举 ~0.7 s/步）
和**升级率**（实测 32–36%，卡在 TRAINING §7 的 35% 及格线上）。
另修掉一个会让 τ 语义失效的校准接入 bug（见 §4）。

## 1. 这次动了什么

| 文件 | 改动 | 为什么 |
|---|---|---|
| `screen_use/jev_bridge.py` | **新增**：OPS 单一真源 + `to_action()` + `history_entry()` + `load_engine()` | 运行时包不背 torch；train/serve 的词表与 history 文本格式必须同源 |
| `screen_use/agent.py` | `VisualLoop(jev=...)` 快路径：`_observe()` 轻观测 → Jev 决策 → 低置信/不可执行才落 VLM；BLOCKED 直接升级；`Step.ring`/`metrics` | 协议位阶段 A 已留好，这里把中环真正插进去 |
| `screen_use/tools.py` | `ScreenUse(jev=...)` + `JEV_CHECKPOINT` 懒加载；`run_task` 透传；`EscalationResult` 增 `engine`/`metrics`/`from_ring=jev`/`blocked_action` | 宿主需要知道这一步是谁开的车、以及该不该再试快路径 |
| `screen_use/config.py` | `jev_checkpoint`（默认空 = 纯 VLM，行为与接入前完全一致） | 可回滚、可对照 |
| `training/serve.py` | ① TYPE 的 **goal-copy 文本**（span 头解码）② **等张校准置信度**（τ 的语义对齐）| ① 不补就没法执行 TYPE 步 ② 原实现用原始 sigmoid 比校准后选的 τ → 升级率漂移 |
| `training/calibrate.py` | 留出集**再分半**（半拟合、半选 τ + 报 ECE）；等张断点落盘 | 原实现同批拟合+评估 → ECE=0.0 是自欺数字 |
| `training/common.py` | `OPS`/`_hist_entry` 反向复用 bridge | 单一真源 |
| `screen_use/memory.py` | `TraceRecorder.record_step(source=...)` | 轨迹要能区分「谁开的车」（TRAINING §1 分层报告） |
| `scripts/bench_jev.py` | **新增**：无头 replay 北极星测试（真 Jev vs 真 VLM，同一步同一观测） | mock GUI 量产受「owner 在机」限制，速度对比的自变量是决策耗时，可用 gold 轨迹复现观测 |
| `scripts/eval_offline.py` | **新增**：多切片离线评测 + **部署侧** selective 曲线 | 单切片 joint_acc 能差 10 个点；τ 必须在部署路径上选 |
| `scripts/smoke_jev_loop.py` | **新增**：真桌面集成冒烟（dry_run + 桩 VLM + 零焦点） | 验证接线在真机上不炸，且不抢鼠标/不留脏轨迹 |
| `scripts/bench_gui.py` | **新增**：真 GUI 端到端（mock 窗口 + 零鼠标执行器 + 真 VisualLoop），默认拒跑、需 `--allow-gui` | owner 离开电脑时一条命令拿真墙钟；`--offscreen` 可随时自检代码路径 |
| `tests/` | 118 → **155 passed**（新增 bridge 一致性、快路径/升级/认输/异常降级/指标） | 回归护栏 |

## 2. 北极星首测（无头 replay，2026-09-19 10:37）

同一批 mock gold 轨迹的**同一步、同一元素表、同一 goal**，分别交给两个驾驶者：

| 指标 | Jev（0.6B，GPU） | VLM（kimi-for-coding，云） | 倍数 |
|---|---|---|---|
| 决策时延 p50 | **43 ms** | 5 978 ms | **139×** |
| 决策时延 mean / p95 | 47 / 61 ms | 6 200 / 12 072 ms | ~130× |
| op_acc（对 oracle） | 0.983（n=173） | 0.90（n=10） | — |
| joint_acc（op+元素+文本） | 0.769 | 0.90 | — |
| 每步是否要截图 | 否 | 是（+ SoM 编码） | — |

端到端推算（每任务，同 4 个回合配对）：

| 驾驶方式 | 每任务 | VLM 调用/任务 |
|---|---|---|
| 纯 VLM（现状对照组） | 17 719 ms | 2.5 |
| **Jev 快路径 + 升级** | **10 190 ms（1.74×）** | 1.5 |

拆分（每步）：观测 0.68 s（UIA 枚举，两个模式都要）+ 执行 0.18 s + 决策（Jev 0.05 s 或 VLM 6.2 s）。
**升级的步要付两份观测**（先 UIA 再补 SoM 截图），所以升级率直接决定混合模式还剩多少优势。

## 3. 离线与部署侧指标（`offline_eval.json`）

| 指标 | 值 | 说明 |
|---|---|---|
| op_acc（3 个 10% 切片） | 0.962 ± 0.018 | 与训练日志 0.945 一致量级 |
| joint_acc（同上） | **0.758 ± 0.061** | 切片方差极大 —— 单切片报数字不可信 |
| ECE（校准前） | 0.071 | 温度缩放 + 等张后见下 |
| 部署侧 coverage @ τ=0.77 | **0.677**（n=300，等距抽样） | 校准文件里同一 τ 写的是 0.759（10% 切片，偏乐观） |
| 部署侧升级率 | **0.323** | TRAINING §7 及格线 ≤0.35，压线通过 |
| 部署侧 op_acc / joint_acc | 0.970 / 0.793 | 与离线切片同量级 |
| 推荐 τ（覆盖率目标 72.5%） | **0.72** | 部署曲线给的是 0.743 覆盖率 |

**τ 已按推荐值钉到 0.72**（`training.calibrate --tau 0.72`，`calibration.json` 标 `tau_source: cli`），
改后重新在部署路径上复核（n=300）：**覆盖率 0.743 · 升级率 0.257 · 快路径风险 0.103** ·
op_acc 0.970 / joint_acc 0.793 · 单步 p50 38.9 ms（空闲 GPU）。
注意 `checkpoints/` 在 `.gitignore` 里，τ 不在版本控制内。

## 4. 修掉的坑：τ 的语义在部署侧丢了

- 现象：`calibration.json` 的 τ=0.5 是在**等张校准后**的 conf 上选出来的，
  但文件里没存等张映射，`serve.py` 拿**原始 sigmoid** 去比 τ。
- 后果：离线「覆盖率 70.9%」在线上根本不是那个数，升级率/VLM 调用数全部失控。
- 修法：`calibrate.py` 落盘 isotonic 断点 → `serve.py` 用同一套分段线性插值；
  `calibration.json` 里没有 isotonic（旧 ckpt）时如实回退原始 conf，并用 `confidence_raw` 保留诊断值。
- 同时把留出集**按轨迹文件再分半**：一半拟合温度/等张，一半只选 τ + 报 ECE
  （原实现同批拟合+评估，ECE=0.0 是假象）。
- 旧 ckpt 重校准结果：`temperature_op 0.7 / temperature_elem 0.8 / τ 0.77 /
  覆盖率 0.759 / 风险 0.069 / ECE 0.095→0.067（选择集）`。

## 6. 观测成本攻坚（首测后立刻做的一件事）

**体检结果**（`scripts/measure_observe.py`，本机实测，纯只读）：

| 项 | 值 |
|---|---|
| UIA 枚举总耗时 | **0.97–1.6 s/次**（前台窗口 + 后台窗口 DFS） |
| 访问节点 | 437 个（预算 3000，远没到顶） |
| 产出元素 | 51–56 个 |
| 顶层窗口收集 | 133 ms |
| **两个无关后台窗口** | 音乐播放器 ~240 ms + Alienware 控制中心 ~280–450 ms（**占 86%**） |
| 前台窗口（终端） | 47 节点 / 53 ms |
| 截图 + SoM + JPEG | ~120 ms（比枚举便宜得多） |

根因：`uiautomation` 的 `GetChildren()` 是 `GetFirstChild → GetNextSibling` 逐节点走，
每个节点还要读 `ControlTypeName`/`BoundingRectangle`；慢 provider 每节点 ~4 ms 的跨进程
COM 往返。**节点数不是瓶颈，每节点成本才是。**

试过三条路：

| 方案 | 结果 | 结论 |
|---|---|---|
| ①多窗口并行遍历（线程各自 CoInitialize + 认领窗口） | 1.49 s vs 串行 1.52 s，元素表完全一致 | **无收益**，已回退（UIA 跨进程调用在 broker 侧仍串行） |
| ②枚举时间预算（前台不限时 + 后台按历史成本从便宜到贵走，超时即停） | 400 ms → 观测 0.55 s / 27 元素；150 ms → 0.18 s / 17 元素 | **有效但有代价**（元素表变短），做成开关 `SCREEN_USE_UIA_BUDGET_MS`，**默认 0=不限**（不偷偷改变行为） |
| ③升级步复用刚枚举的元素表（`screenshot(elements=...)`） | 每次升级省一整轮枚举（0.5–1.5 s），零风险 | **已默认开启**（`metrics.enum_reuse` 计数） |

**还差的一步（设计已定，下个会话做）**：后台窗口元素表**缓存** ——
按 `(窗口句柄, 标题, 矩形)` 命中则复用，前台窗口照旧每步重走；
点之前用控件对象校验实时 bbox 与缓存是否一致（不一致→刷新重解析 id）。
预期：每步 1.3 s → ~0.2 s，且**元素表不缩水**。这是观测路径上剩下的最大一块，
但要动到点击路径，必须配 GUI 实测（不能盲改）。

## 7. 诚实边界

1. **VLM 侧样本只有 10 步**（按 API 预算抽的 4 个回合）。它的时延量级稳（p50 6.0 s），
   准确率数字噪声大，不能当结论用。
2. **VLM 看的是合成底图**（灰白窗口 + 控件名文字），编号框用的是线上同一份 `som.annotate`，
   prompt 与 `VisualLoop` 逐字符一致。合成底图让 VLM 吃亏，不影响 Jev（它不看图）。
3. **观测/执行成本是「本机实测 + 常量」**：UIA 枚举 0.68 s 是在 owner 当前桌面（多窗口）实测的；
   执行 0.18 s/步是 mock UIA Invoke 的常量，两个驾驶者同值、在对比中抵消。
   真实 mock 窗口里的 UIA 枚举通常更快。
4. **这不是真 GUI 端到端**（无头 replay）。真 GUI 需要开窗，按约束只能 owner 离开时跑：
   `scripts/bench_gui.py --allow-gui --driver {oracle,jev,vlm}` 已备好（零鼠标执行器 + 真
   `VisualLoop`），代码路径已用 `--offscreen` 自检（窗口/截图/标注/动作分发/清理都能跑通），
   但**真窗口 + 真 UIA 那条路没跑过**——首次运行建议人在旁边。
5. **R4 校准漂移仍未验证**：`traces_full` 的 105 步真实轨迹在 `iter_samples` 里几乎全被
   过滤（hotkey/press 不在动作集内），所以部署侧 coverage 只在 mock 域上量过。
6. **SCROLL/WAIT 没有载荷**（v0 动作集只有 8 类，无格数/秒数头）→ 快路径用默认值
   （滚动 -3、等待 1s），属已知精度损失；mock 里 scroll 步占比低，首测影响小。
7. **BLOCKED 会提前结束任务**：mock 首测里认输 0 次（命中 0、误认输 0），
   但真实桌面冒烟时 Jev 曾主动低置信升级——认输的假阳性代价需要更多真实数据才能估。

## 8. 下一步建议（按性价比）

1. **压升级率**：τ 已钉到 0.72（部署复核覆盖率 0.743 / 升级率 0.257），零训练成本。
2. **后台窗口元素表缓存**（见 §6）：观测路径剩下的最大一块，预期 1.3 s → 0.2 s 且不缩水。
3. **真 GUI 端到端**（owner 离开时跑）：`scripts/bench_gui.py`，拿真墙钟与真实升级率。
4. **SFT v1**：joint_acc 0.758 距离 ≥0.80 的及格线还差一点；1.7B 基座 + 真实轨迹能补。
5. **量化导出**：p50 38.9 ms 已达标（<100 ms），ONNX/int8 可再压一档，
   但观测不解决的话收益会被观测吃掉。
