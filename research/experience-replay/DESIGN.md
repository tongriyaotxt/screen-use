# 经验回放（Experience Replay）设计文档

> 2026-08-27 · 配套阅读：`literature.md`（文献证据）、`gaps.md`（空白分析）
> 状态：设计提案，未实现。对应代码基线：screen-use @ 速度改良版（103 tests）。

## 0. 一句话

把 VLM 视觉循环**探索成功**的任务轨迹，编译成**不依赖模型的参数化宏（skill）**——语义锚点定位、检查点验证、失败即回退 VLM 循环——让重复任务的模型调用从 N 次降到 ~3 次。

形式化对应：宏 = options framework（Sutton 1999）的 option = ⟨initiation set, policy, termination⟩：
语义锚点 ≈ initiation set，动作序列 ≈ policy，检查点 ≈ termination condition，回退 VLM ≈ meta-policy 重新选择。

## 1. 为什么是现在

证据链（详见 literature.md）：

- **宏替代逐步推理有直接量化先例**：AppAgentX（2025）把常用动作序列进化为高层动作后，成功率 46.3%→88.2%，每步 43.5s→17.5s，token 11.5k→5.1k。
- **"技能=确定性代码"已验证**：Voyager 技能库（消融证明 self-verification 去掉掉 73%——验证机制不是配件，是核心）。
- **现有经验复用全停在 prompt 级**（AWM/Synapse/AutoGuide：检索进 prompt，每步仍调 LLM）。**"执行级"复用 + Windows/UIA 域是文献空白**（gaps.md §1a/§1b）。
- **我们的零件已齐**：`_uia_match`（语义锚点）、`do_actions`（回放引擎）、`elements_fingerprint`（检查点比对）、`ExperienceStore`（轨迹存储）、`memory.recall`（任务检索）、VisualLoop（兜底慢层）。

## 2. 核心设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 宏存什么 | **语义锚点** `(window_title, control_type, name)`，不存坐标/id | id 每轮重排、坐标随窗口漂移；UIA 名称稳定。这是与 RPA 1.0 selector 的本质区别 |
| 何时验证 | **检查点稀疏验证**（窗口切换后、文本写入后、收尾前），非每步 | 每步验证=回到逐轮截图的老路；AppAgentX 证明粗粒度动作足够 |
| 失败怎么办 | **bailout**：回放中止，从当前真实状态交给 VisualLoop 续跑 | 下行有界：最坏 = 浪费几秒 + 和今天一样。这是组合方案无先例但最关键的一环（gaps.md §1c） |
| 参数怎么来 | 录制时离线 LLM 分析一次 + 启发式（输入文本出现在 goal 中→参数） | 一次性成本，不在热路径 |
| 何时不可用 | 动态内容任务（"总结文档"）、UIA 盲区应用（纯坐标轨迹不回放） | 诚实边界：只覆盖重复 + 结构稳定任务，但那正是 RPA 价值所在 |

## 3. 数据模型

```json
{
  "skill_id": "sha1(goal 模板 + 锚点序列)",
  "goal_template": "在{app}中提交{document}",
  "params": {
    "document": {"source": "goal", "type": "text", "example": "report.pdf"}
  },
  "created_from": "trace hash", "created_at": "...",
  "stats": {"replay_count": 0, "replay_success": 0, "last_drift_at": null},
  "precondition": {"foreground_title_contains": "Papercept"},
  "steps": [
    {"kind": "action", "action": "click_element",
     "anchor": {"window_title": "Submission", "control_type": "HyperlinkControl", "name": "Submit new paper"},
     "checkpoint_after": false},
    {"kind": "action", "action": "set_element_text",
     "anchor": {"control_type": "EditControl", "name": "Title"},
     "text": {"param": "title"}},
    {"kind": "checkpoint", "expect": {"elements_contain": ["Upload"], "foreground_title_contains": "Upload"}},
    {"kind": "wait_stable", "timeout": 5}
  ]
}
```

要点：
- `anchor` 回放时经 `list_ui_elements` + `_uia_match`（精确→模糊）**重新解析为当前元素**，点击用重新解析出的实时坐标——UI 挪动不敏感，改名才失效。
- 锚点解析失败 = 立即 bailout（不是错误，是"环境已漂移"信号）。
- `stats` 支撑退役机制：回放成功率低于阈值（如 3 次内 2 败）自动降级该 skill，强制回到 VLM 探索并重新录制。

## 4. 三条流水线

### 4.1 录制（离线，任务成功后触发）

```
VisualLoop 成功轨迹 (ExperienceStore 已有)
  → 过滤：丢弃失败步骤/think_error/无效 wait
  → 锚点化：click_id → 该元素的 (window_title, control_type, name)
            type    → 前置 click 的锚点 + 文本（尝试转 set_element_text，更快更稳）
  → 参数提取：一次离线 LLM 调用分析轨迹 + goal，标出数据参数（启发式：文本 ⊂ goal）
  → 检查点插入：窗口标题变化处、type 之后、收尾前
  → 入库前自验证（Voyager 的教训：无验证入库 = 污染技能库）：
    干跑一遍回放，全部检查点通过才标记 verified
```

### 4.2 检索与触发（热路径，零模型）

`run_task(goal)` 入口改为：

```
1. memory.recall_skill(goal)           # 现有 recall() 扩展：goal 相似度匹配 skill
2. 检查 precondition（前台标题/应用名）   # 不满足 → 直接走 VLM 循环
3. 命中 verified skill → 进入回放；未命中 → VLM 循环（现状）
```

### 4.3 回放执行器（复用 do_actions 的分发逻辑）

```
for step in skill.steps:
    action:   锚点 → list_ui_elements → _uia_match → 执行（元素缺失 → bailout）
    checkpoint: elements_fingerprint / 元素名存在性 / 前台标题 比对（失败 → bailout）
    wait_stable: 轮询指纹稳定（替代固定 sleep，顺便实现智能等待）

bailout(reason):
    记录 drift 事件（skill.stats）
    把 goal + 已完成的宏步骤摘要 + bailout 原因 注入 VisualLoop 初始 history
    → VLM 从当前真实状态接管续跑（不是从头再来）
```

## 5. 与现有代码的对接面（最小侵入）

| 新模块 | 复用 | 改动 |
|---|---|---|
| `screen_use/skills/`（新包：schema / compiler / replayer / store） | — | 新建 |
| `ExperienceStore` | 存 skill JSON | 加 `record_skill/recall_skill/update_skill_stats` |
| `agent.VisualLoop.run` | 兜底慢层 | 入口加 skill 检索分支；接受 `resume_context`（bailout 续跑） |
| `tools.ScreenUse` | `_uia_match`、`_activate_window`、`do_actions` | `_uia_match` 从静态方法提为模块函数供复用 |
| `reflect.elements_fingerprint` | 检查点比对 | 不动 |

预估：MVP ~400 行新代码 + 测试。现有 103 测试不受影响。

## 6. 实施路线

**Phase 1（分段回放，风险最小）**：只编译轨迹中的"确定性中段"（连续 ≥4 个锚点动作，无坐标 click、无 scroll），导航/决策仍归 VLM。验证锚点机制和 bailout 协议。
**Phase 2（完整宏）**：参数化 + precondition + 自验证入库 + 退役机制。
**Phase 3（量化评估，可发表点）**：度量"回放覆盖率 = 回放步骤 / 总步骤"与"VLM 调用次数/任务耗时"的关系曲线——gaps.md §3 指出这个量化在文献中是空白。

## 7. 评估指标

- **回放覆盖率**：回放完成步骤占比（目标：稳定重复任务 >80%）
- **VLM 调用次数**：同一重复任务第 1 次（探索）vs 第 2+ 次（回放）
- **bailout 率与原因分布**：漂移检测的灵敏度校准依据
- **端到端耗时**：对标 AppAgentX 的 43.5s→17.5s/step，预期重复任务总耗时降 5-10×
- 回归基准：固定任务集（计算器、记事本、投稿表单 dry-run）CI 可跑

## 8. 具身智能迁移路径（长期目标）

**主张**：单纯 VLA 端到端控制是低效的——RT-2 仅 1-3Hz、OpenVLA 7B 在 4090 上 6Hz、π0 仅 KV prefill 就 46ms 超出 50Hz 控制周期 20ms 的预算（literature.md §3）。推理延迟只随模型变大而增加（RTC 论文），**异步分层是必然架构而非权宜之计**。Hi Robot / GR00T N1 的快慢分层已是社区主流。

**本项目的迁移论点**：GUI 回放宏 = 快慢分层中"快层"在**离散可枚举状态空间**的特例——
- GUI 状态可用 UIA 树枚举、指纹可精确比对 → 快层可以是**零模型的确定性回放**
- 物理世界连续、部分可观测 → 快层需要 learned policy（π0 阵营的立场）或模型预测控制

**从 GUI 到具身需要换掉的零件**（接口不变）：

| GUI（screen-use） | 具身（迁移目标） |
|---|---|
| UIA 语义锚点 | 视觉 grounding（UGround 类小模型）/ 物体位姿 |
| elements_fingerprint 检查点 | 状态估计 / 触觉-视觉一致性检查 |
| bailout → VLM 循环 | bailout → VLA 慢层重新规划 |
| 宏 = 动作序列 | skill = 动作 chunk 序列 / 运动基元 |

**诚实的学术风险**（gaps.md §3）：现有快慢层的快层**全是 learned policy**，"快层=回放宏"在具身域是主张而非结论；GUI 域可假设重复任务状态可枚举，物理世界不行。因此论文/叙事定位应为：**在 GUI 域先量化"回放能替代多少大模型调用"（空白点），再论证该比例随环境可枚举性衰减的规律**——而不是直接主张具身快层该用回放。

**战略价值**：screen-use 的每一次真实任务都在积累"探索轨迹 → 技能"的转换数据。这套 pipeline（探索-蒸馏-回放-退役）本身与本体无关，是迁移时唯一需要完整带走的东西。
