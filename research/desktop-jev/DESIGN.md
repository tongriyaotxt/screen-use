# 桌面版 Jev（desktop-jev）设计文档

> 2026-09-18 · 配套阅读：`literature.md`（文献证据）、`gaps.md`（空白分析）
> 状态：阶段 0（轨迹采集）已实现（2026-09-18，111 tests）；其余为设计提案。对应代码基线：screen-use @ desktop-jev 阶段 0 版（111 tests）。
> 关联：`../experience-replay/DESIGN.md`（三级 cascade 的第零层）。

## 0. 一句话

训练一个 0.5B–1.5B 的**校准决策模型**（Jev 架构思路：非自回归、类型化输出头、校准置信度），
接管 VLM 视觉循环中占大头的**定位/导航决策**——点哪个元素、滚不滚、等不等、完成没——
**文本生成与低置信步骤升级回 VLM**。定位是加速器，不是替代者。

形式化对应：三级 cascade 是 Kahneman System 1/2 在 GUI agent 上的实例化——
宏（experience-replay）≈ 肌肉记忆（零模型），desktop-jev ≈ System 1（快模型，校准置信度），
VLM 循环 ≈ System 2（慢模型，全能力兜底）。升级协议 = metacontrol：
由 System 1 自报的校准概率决定每一步给谁执行。

## 1. 定位约束（owner 裁定，2026-09-18）

> "小模型部分代替 VLM/大模型，主要是为了加速；不能输入文字，不能完全替代。"

由此推出的架构根因：

- **输出端砍死自由文本生成**。非自回归并行解码与字符串生成互斥（Jev 同理：
  "gives up string generation"）。任何需要"写出一段文字"的步骤（总结、回复、
  长表单自由撰写）天然不在小模型能力域内，必须升级。
- **覆盖率目标不是 100%，而是"定位步占比"**。GUI 任务轨迹中，点击/滚动/等待/判断完成
  类步骤占大头（文献中 Web agent 轨迹的 click/scroll 占比通常 >60%，见 literature.md），
  小模型吃掉这部分即可把 VLM 调用从"每步一次"降到"仅文本步 + 困难步"。
- **预期 VLM 调用降 70–90% 是待验证假设**，依赖两个未测量量：
  ① 真实任务中无文本步占比；② 小模型在桌面观测上的可达准确率。第一公里（轨迹采集）就是为测量它们。

## 2. 为什么是现在

证据链（详见 literature.md）：

- **Jev 验证了 System One 决策模型的产品形态**（TypeSafe AI，2026-09-15）：
  非自回归并行采样、三种类型化原语（Choice/Score/Noul）、校准置信度、
  70–500ms 端到端、$0.042/MTok 输入且输出免费。CEO 是 InstructGPT/RLHF 作者 Diogo Almeida。
  注意：Jev 是闭源 API、纯文本输入、无图像；**它证明了架构范式，不提供桌面可用工件**。
- **RLCD 的核心洞见可直接借用**：把训练目标从"答对"换成"校准"（说 70% 把握就真的 70% 对），
  使"置信度门控升级"成为可靠的系统原语。我们对 RLCD 的平价替代：
  SFT + temperature scaling（Guo et al. 2017）+ 留出集等张回归，文献成熟、零额外 rollout。
- **我们的零件已齐**：`list_ui_elements`（≤150 元素的结构化观测，桌面版 DOM 快照）、
  `VisualLoop._ACTIONS`（动作空间已基本固定）、`_dispatch_action`（9 种动作的执行层）、
  `ExperienceStore`（轨迹存储，待升级为结构化逐步采集）、
  experience-replay 的 bailout 协议（升级机制的镜像先例）。
- **GUI agent 的"快层"真空**：UI-TARS/Aguvis/ShowUI 等桌面 agent 全是
  "每步一次大模型前向"，无一采用校准小模型做步级路由（gaps.md §1a）。

## 3. 核心设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 模型角色 | **步级决策器（fast path），非端到端 agent** | owner 定位约束；与 Jev 的"function call"形态一致：状态进、类型化决策出 |
| 观测 | **UIA 元素表序列化文本**，不走截图 | `UIElement.to_dict()` 即桌面版 DOM 快照；ScreenSpot-Pro 证明纯视觉 grounding 在桌面专业软件上仅 ~19–48%，UIA 是桌面域独有杠杆；零图像 token = 上下文小、推理快 |
| 动作集 | **收窄子集**：CLICK / DOUBLE_CLICK / RIGHT_CLICK / SCROLL / WAIT / DONE / BLOCKED + 受限 TYPE | 覆盖循环大头；TYPE 文本只允许 **goal-copy 槽位**（从 goal 中指 span），自由文本判 BLOCKED 升级 |
| 输出头 | 操作分类头 + 元素指针头 + 置信度头 + 文本槽位指针头 | 全部单次前向并行输出，无解码循环；元素指针在候选表上做 softmax（≤150 类），天然类型安全 |
| 升级判据 | **置信度阈值 + 动作类型白名单 + 停滞检测**三条件任一触发 | 校准是 RLCD 路线的核心资产；停滞检测复用 `classify_stuck`（reflect.py） |
| 升级事件处理 | **自动进标注队列**（困难样本） | 升级点即分布边界，正是 API/教师预算最该花的地方 |
| 与宏层关系 | 宏优先、小模型其次、VLM 兜底；三者只增不改 | 重复任务零模型；新任务快路径；困难/文本步慢路径。每层失败/低置信都向下交权 |

## 4. 观测空间

单步观测 = 任务描述 + 前台窗口标题 + 元素表 + 近期动作历史，序列化为纯文本：

```
TASK: 在计算器算 78*9 并复制结果到记事本
WINDOW: 计算器
HISTORY: [1] CLICK(#12 "Seven") ok | [2] CLICK(#34 "Multiply by") ok
ELEMENTS:
#1 [ButtonControl] "One" bbox=(412,330,446,352) win="计算器"
#2 [ButtonControl] "Two" ...
...（≤150 条，面积升序，与 list_ui_elements 当前输出一致）
```

token 预算估算（诚实标注：数量级估算，待实测）：

| 组件 | 估算 |
|---|---|
| 元素表 150 条 × ~20 token | ~3,000 |
| TASK + WINDOW + HISTORY（近 8 步） | ~300–600 |
| **合计** | **~3.5–4K，留 32K 上下文的一半以下** |

压缩手段（超预算时按序启用）：bbox 坐标整数化已自带；名称截断 32 字符；
HISTORY 滑窗；元素配额按前台窗口倾斜（现有 `MAX_ELEMENTS=150` 配额逻辑即为此设计）。
不可压缩项：control_type（类型先验对决策头价值最高，UGround/OS-Atlas 均利用之）。

## 5. 动作空间与输出头

### 5.1 动作集（8 类，对齐现有执行层）

| 动作 | 参数 | 小模型可决？ | 映射到现有代码 |
|---|---|---|---|
| CLICK | element_id | ✅ | `click_element_id` / `_act("click_id")` |
| DOUBLE_CLICK / RIGHT_CLICK | element_id | ✅ | `do_actions` 同名动作 |
| SCROLL | clicks(±), 可选 element_id | ✅ | `scroll` |
| WAIT | — | ✅ | `_act("wait")` |
| DONE | — | ✅（完成判断是校准问题） | `_act("done")` |
| BLOCKED | — | ✅（主动认输=升级信号） | 新增：触发升级协议 |
| TYPE | element_id + **goal-copy span** | ⚠️ 受限 | `set_element_text`；span 越界/无槽位 → BLOCKED |
| （自由文本输入） | — | ❌ 永升级 | VLM 决定文本后仍可走 `set_element_text` |

现有 `HOTKEY/PRESS` 不进小模型动作集（低频且语义模糊，直接划给 VLM）——收窄类别数是提升分类头准确率的最便宜手段。

### 5.2 输出头（单次前向并行）

```
输入嵌入（≤4K token）
   ├─ 操作头：softmax over 8 类
   ├─ 元素头：pointer-softmax over 150 候选 + 1 个 "no-element" 槽（SCROLL/WAIT/DONE 用）
   ├─ 槽位头：goal token 序列上的 start/end 指针（copy 机制，仅 TYPE 激活）
   └─ 置信度头：标量 σ（与操作头分布联训，RLCD 精神：置信度 = 校准后的正确概率）
```

基座选型：Qwen3-0.6B / 1.7B（Apache-2.0，2025-04）。砍 lm_head 换四头。
5070Ti（16GB）实测预算：1.7B bf16 全量微调 + 8bit optimizer + gradient checkpointing ≈ 12–14GB，可行；
QLoRA 更宽松但没必要（分类头任务容量要求低）。

## 6. 升级协议（核心机制）

插在 `VisualLoop.run` 的 observe 与 think 之间（`_think` 前），**纯增量、不改慢路径**：

```
per step:
  1. obs = 元素表 + goal + history        # 复用本轮 observe 结果，零额外截图
  2. (action, elem, conf) = jev(obs)      # 单次前向，目标 <100ms（本地 1.7B）
  3. if action == BLOCKED
       or conf < τ_auto                   # τ 在留出集上按 ECE 调
       or action 不在白名单需求内          # 需要自由文本
       or stuck_detector 报警:            # 复用 classify_stuck
  4.     → 升级 VLM（走现有 _think），本步观测+小模型预测+升级原因写入标注队列
  5. else → 直接 _act(action, elem)，跳过 VLM 调用
```

与 experience-replay bailout 的对偶关系：bailout 是**确定性执行失败后**被动回退；
升级协议是**快路径决策前**主动交权。两者共享同一慢层（VisualLoop）与同一原则——
**下行有界：最坏情况 = 浪费一次 <100ms 前向，行为与今天完全一致**。

τ 的调法（temperature scaling 之后）：在留出集上选 τ 使"升级率 ≤ 预算（如 25%）"前提下
快路径准确率最大化；上线后按滑动窗口升级率漂移再校准。

## 7. 三级 cascade 全景

```
run_task(goal)
  ├─ L0 宏层（experience-replay，未实现）：命中 → 零模型回放，检查点失败 → L2
  ├─ L1 desktop-jev（本设计）：步级快决策，BLOCKED/低置信/需文本 → L2
  └─ L2 VLM 视觉循环（现状）：全能力慢路径，产出的轨迹反哺 L0/L1 数据
```

量化目标（**待验证假设**，非承诺）：

- 重复任务：L0 覆盖，~3 次模型调用/任务（experience-replay DESIGN 已论证）。
- 新任务：L1 覆盖无文本步，VLM 调用从 N 次/任务降到"文本步 + 困难步"次数。
- 合并预期：整体 VLM 调用降 70–90%。依赖测量：无文本步占比、L1 准确率、L0 命中率——
  全部由第一公里的轨迹采集提供。

## 8. 数据策略（三条腿）

桌面没有 MiniWoB++。按成本从低到高：

| # | 来源 | 机制 | 量级目标 | 成本 |
|---|---|---|---|---|
| 1 | **真实轨迹采集** | VisualLoop 跑真实任务时逐步落盘 `(元素表, VLM决策, ok)` | 先攒 1–5K 步 | 零（副产品） |
| 2 | **mock 应用合成** | tkinter/Qt 程序化生成窗口+任务，oracle 策略给 ground truth | 10K+ 步 | 零模型成本，工程量中 |
| 3 | **教师标注** | 本地 Qwen3-8B/14B AWQ（5070Ti 可跑）标注腿 1/2 覆盖不到的难例；两本地教师分歧才烧 API | 1–3K 步 | 极低 API |

- 腿 1 的改造点：`ExperienceStore` 目前只存摘要（goal/action/args/ok），**不存每步元素表**——
  无法用作 SFT 输入。需升级为结构化逐步采集（独立工程增量，见 §10 阶段 0）。
- 腿 2 是桌面域的 MiniWoB++ 等价物：程序化窗口（随机布局的按钮/输入框/列表 +
  模板任务"把 X 填进 Y 并提交"），正确动作天然已知。tkinter 优先（stdlib，UIA 可见性已验证）。
- 腿 3 的 committee 采样省钱法是经典做法：两个本地量化教师一致 → 免费标签；
  分歧 → 送强 API 仲裁。升级协议（§6）捕获的困难样本也走这条路。

## 9. 训练与评测

**训练**：SFT（交叉熵，四头加权和）→ 留出集 temperature scaling + 等张回归校准置信度头。
RLCD 式校准 RL 列为可选后续：若 MiniWoB++ 式合成环境（腿 2）带程序化 verifier，
可小规模 GRPO 补强置信度头，环境 rollout 免费，不烧 API。

**评测指标**（离线→在线）：

| 层 | 指标 | 目标参考 |
|---|---|---|
| 离线 | 操作分类准确率 / 元素指针 top-1 | 先立基线再谈目标；Jev 的参照是"追平大模型" |
| 离线 | 校准 ECE、Brier | temperature scaling 后 ECE < 0.05（Guo 2017 方法可达量级） |
| 离线 | 单步延迟（5070Ti，1.7B，4K ctx） | <100ms（对 VLM 的 3–30s 是 30–300×） |
| 在线 | 真实任务集端到端成功率 / 升级率 / VLM 调用数 | 对照组 = 现状纯 VLM 循环 |

评测集：自建 200–500 条真实桌面任务（标注"下一步操作+元素"），
外加 WindowsAgentArena/OSWorld 子集做外部参照（注意：它们的 harness 重，
只取其任务与 verifier 思路，不自建虚拟机群）。

## 10. 实施路线

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| 0 轨迹采集 | `ExperienceStore` 升级结构化逐步落盘（本设计的唯一前置工程） | ✅ 已完成 2026-09-18：`TraceRecorder` + VisualLoop 钩子 + `record_full_trace` 开关，111 测试全绿 |
| 1 数据基线 | 腿 1 攒 1K+ 步；统计无文本步占比、动作分布 → **验证 §1 覆盖率假设** | 🔄 首批 105 步（2026-09-18）：保守覆盖率 73.8%、乐观 90.5%（`scripts/analyze_traces.py`），假设方向成立；样本量小需持续采集 |
| 2 mock 合成 | tkinter 程序化窗口 + oracle 策略，10K 步 | ✅ 首版完成 2026-09-18：实际用 PySide6（tkinter 按钮无名/输入框不可见，UIA 不可用）。6 类模板（click/type/form/list/scroll/blocked），`scripts/mock_env.py`，17/17 oracle 执行成功 |
| 3 SFT v0 | Qwen3-0.6B 四头，腿 2 数据 + 腿 1 微调 | 离线指标准确率/延迟基线 |
| 4 校准 + 接入 | temperature scaling；§6 升级协议接入 VisualLoop | 真实任务集：升级率、VLM 调用数、成功率三指标齐测 |
| 5 教师蒸馏 | 腿 3 committee 标注难例，SFT v1 | 困难子集准确率提升 |

## 11. MCP 形态下的提速路径（2026-09-18 owner 提问后补充）

screen-use 的主要使用形态是 MCP server——循环在**宿主 agent 脑内**，不在我们的 VLM。
这改变了 desktop-jev 的收益结构：

| 位置 | 现状 | Jev 后 | 收益量级 |
|---|---|---|---|
| `find_element`/`click_element` VLM 兜底 | UIA 匹配失败 → SoM+VLM pick_element（3–10s） | 小模型 Choice-over-elements（<100ms），低置信升级 | 每次调用省 3–10s |
| `run_task`（也是 MCP 工具） | 每步一次 VLM | 73.8–90.5% 步零 VLM（阶段 1 实测） | 主战场，同 VisualLoop |
| 动作后快验证（潜在新 MCP 工具） | 宿主再截图+思考一轮 | Noul 式快检（<100ms） | 省一轮宿主往返 |
| 宿主 LLM 规划时间 | 每步 3–30s | **管不着** | — |
| 聪明宿主 + UIA 元素表直驾 | 本就零 VLM | 无加速空间 | — |

结论：MCP 形态下 Jev 的定位从"加速循环"修正为"**消灭工具内部的 VLM 调用**"。
且 MCP 形态下更大的提速杠杆是 experience-replay（宏作为 `run_skill` 工具，
一次调用顶宿主十几轮往返，确定性零模型）。**MCP 形态优先级：宏 > Jev**；
Jev 的实现顺序不变（数据先行），但首个接入点除 `run_task` 外增加 `find_element` 策略链。

## 13. 路线复盘（2026-09-18 owner 战略提问后）

**Jev 的本质**：把感知-动作映射从 System 2 压缩成 System 1。宿主=脑（任务规划），
Jev=手（反射执行）。GUI 操作 74-90% 的步是反射不是推理（阶段 1 实测），
这就是"比人快"的理论依据。

**体检结论**：UIA 观测 ✅、决策模型化 ✅、校准升级 ✅（有坑见 R4）、协议先行 ✅、
mock 定位 ⚠️ 已修正。

**mock 的角色修正**：mock 覆盖**交互范式**（按钮语义对抗/状态依赖/弹窗栈/菜单/
树/表格/向导/下拉/虚拟化滚动——可枚举），不追求**场景皮肤**（模拟所有软件的样子
是死胡同）。场景多样性由三条腿重新分工解决：

| 腿 | 解决 | 方式 |
|---|---|---|
| mock | 交互范式 + 语义/顺序先验 | 范式清单全覆盖 |
| 网页 | DOM 密集型 UI | HTML 生成器 + 浏览器 + UIA（网页域 MiniWoB++） |
| 真实轨迹+教师 | 真实软件分布与难例 | 持续采集 |

**新增风险**：
- R4 校准漂移：mock 上校准 ≠ 真实软件上校准 → τ 必须用真实数据在线再校准，写入接入阶段验收。
- R5 UIA 盲区：游戏/自绘 UI 无元素表，v1 全升级（协议兜底不失速）；像素路径（视觉 grounding）属 v2。

## 12. 诚实边界与风险

- **Jev 的全部性能数字来自 TypeSafe 自家 eval**（workflow evals 以 GPT-6 Astra/Fable 5.1
  平均预测为参照而非 ground truth，Hacker News 已有质疑），架构细节未发表。
  本文档只借其**范式**（并行类型化输出 + 校准目标），不借其**数字**。
- **覆盖率假设可能证伪**：若真实桌面任务文本步占比过高，或小模型在 UIA 观测上
  准确率上不去，70–90% 的 VLM 削减不成立。阶段 1 的 kill criterion 就是为此设的。
- **校准不等于正确**：Jev 官方也承认 schema 合法只保证接口不保证真值。
  升级协议把错误决策的代价限制在"一步错误动作"，但桌面动作有副作用（发送、删除），
  不可逆动作应配置为**永远走 VLM 或人工确认**——这是策略层配置，不是模型能解决的。
- **UIA 盲区**：部分应用（自绘 UI、游戏、部分 Electron）元素表为空或不完整，
  此时小模型无观测可吃，天然全升级——与现状一致，无回归。
- **goal-copy 槽位覆盖有限**：密码、动态生成内容、多步拼接文本都超出 copy 机制，
  这类任务全程慢路径，属预期行为。
