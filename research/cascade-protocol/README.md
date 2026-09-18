# 三环契约（Cascade Protocol）：让宿主通过 MCP 比人更快地操作电脑

> 2026-09-18 · 统摄 `../experience-replay/`（内环）与 `../desktop-jev/`（中环）
> 来源：owner 三点裁定——①内环实时更新+执行前适配判断；②写清 Jev 能力方便宿主下指令；
> ③失败退回宿主完全控制。本文是三环对**宿主大模型**的完整契约设计。
> 状态：设计提案，部分零件已实现（标注于 §6）。北极星指标：端到端耗时 < 人类基线。

## 0. 设计原则

1. **内环要可信**：宏执行前必须零模型验证环境适配；宏库实时更新、失败退役。
2. **中环要有说明书**：Jev 能力边界写成 manifest，宿主据此把任务拆成 Jev 友好的形状——
   对 Jev 的要求转化为对宿主下指令方式的引导。
3. **外环要能接管**：失败退回时"带着地图退"——观测快照+已做步骤+失败原因+建议，
   宿主无缝接管而非从零重看。
4. **零模型优先**：能用 UIA 查询判断的（适配、验证），绝不动用模型——否则失去速度意义。
5. **协议先行**：本文的协议层（§3–§5）不依赖任何模型训练，可立即实现并产生价值。

## 1. 三环总览

```
宿主大模型（任务规划层：该做什么）
  │ 启动时读 capabilities manifest（§3），按各环能力拆任务
  ▼
┌─ MCP 工具面（决策路由）──────────────────────────────────┐
│ run_skill(skill, params)   内环·宏：适配判断→回放→检查点   │ 0 模型/步
│   └ 不适配/失败 ──┐                                       │
│ run_task(goal)     中环·Jev：步级决策（0.1s/步）           │ ~0.1s/步
│   └ BLOCKED/低置信 ┤                                      │
│   └ Jev 未训练前：VLM 视觉循环（现状）                     │ 3–30s/步
│                     ▼ 统一 escalate（§4 带地图退）         │
│ 原子工具（click/type/…） 外环：宿主完全控制，逐步驾驶       │ 宿主速度
└─────────────────────────────────────────────────────────┘
         │ 反馈环（§2）：成功轨迹 → 宏候选 → 验证转正 → 入库
```

## 2. 内环：宏的实时更新 + 适配判断

### 2.1 适配判断（执行前，全部零模型）

| 级 | 检查 | 数据源 | 失败动作 |
|---|---|---|---|
| L0 硬前置 | 目标进程/窗口存在、前台标题匹配 `precondition` | UIA 顶层窗口枚举 | 不进内环，落中环 |
| L1 锚点解析 | 宏的全部语义锚点 `(window_title, control_type, name)` 在当前元素表可解析 | `list_ui_elements` + `_uia_match` | 同上 |
| L2 执行中检查点 | 稀疏检查点（窗口切换后、写入后、收尾前） | 元素指纹 `elements_fingerprint` | bailout → 中环从当前状态续跑 |

原则：**三级全是 UIA 查询，一次模型调用都不花**。判断成本 <0.5s，不破坏内环速度。

### 2.2 实时更新流水线（防污染）

```
录制：run_task/run_skill 外的成功轨迹（VLM 或 Jev 产出）→ 离线编译宏候选
hone：宏候选**首次回放必须全程检查点通过**才转正入库（Voyager 消融：验证是核心不是配件）
服役：每次执行更新 stats {replay_count, replay_success, last_drift_at}
退役：近 3 次回放 2 败 → 降权停用，任务回到中环重新探索并重录（宏是有保质期的）
```

- "实时"= 任务成功即触发编译（后台，不在热路径）；执行统计每步更新。
- 宏 schema 沿用 `../experience-replay/DESIGN.md` §3（语义锚点不存坐标）。

## 3. 中环说明书：capabilities manifest

宿主启动时（或需要时）调用 `get_capabilities()`，返回结构化能力清单。
**写清"不能做什么"比"能做什么"更重要**——宿主据此避坑拆任务。

```json
{
  "version": "0.1.0",
  "rings": {
    "skill": {
      "description": "已验证宏的确定性回放，零模型，最快。仅覆盖已录制任务。",
      "list_tool": "list_skills"
    },
    "jev": {
      "description": "小模型步级决策，约0.1s/步。适合：定位/导航/表单填写。",
      "can": [
        "点击/滚动/等待/判断完成（click/scroll/wait/done）",
        "单窗口或明确跨窗口的元素定位与导航",
        "表单填写：值必须显式出现在 goal 中（copy，不创作）"
      ],
      "cannot": [
        "生成自由文本（总结、回复、写作文）——宿主应自己生成文本后作为参数传入",
        "视觉内容理解（读图表、识别验证码）",
        "处理意外弹窗/未知对话框",
        "不可逆操作（发送/删除/付款）——默认需宿主显式确认",
        "多步推理链（'比较A和B再决定'）"
      ],
      "goal_style": "单句祈使句；参数显式给出。例：'在记事本中输入：TODO 周五交报告'，而非'帮我写个备忘录'"
    },
    "atomic": {
      "description": "原子鼠标键盘工具，宿主完全控制。任何任务可用，速度取决于宿主。"
    }
  },
  "escalation": "任何环失败均返回 EscalationResult（见 §4），宿主可无缝接管"
}
```

宿主行为引导（写进 MCP 工具 description，宿主必看）：
1. 有宏用宏（`list_skills` 先查）→ 没宏且任务符合 jev.can → `run_task` → 都不行才原子直驾。
2. 把需要创作的文本**自己生成好**作为参数传入，别让 Jev 写。
3. 长任务拆成 Jev 友好的子任务序列（每个子任务单一目标、参数显式）。

## 4. 外环接管：EscalationResult（带地图退）

任何环无法继续时，**不抛错、不静默**，返回结构化交接包：

```json
{
  "status": "escalated",
  "from_ring": "jev",
  "reason": {"class": "blocked_action | low_confidence | stuck | anchor_lost | precondition_failed",
             "detail": "第4步需要自由文本输入（JEV cannot: 生成自由文本）"},
  "progress": {"steps_done": [{"action": "click_id", "args": {"id": 75}, "ok": true}],
               "remaining_goal": "在原 goal 中尚未完成的部分（尽量给出）"},
  "observation": {"foreground_title": "记事本",
                  "elements": ["...当前完整元素表..."],
                  "screenshot_ref": "可选：MCP image"},
  "suggestion": "建议宿主接管：生成文本后调用 set_element_text(#84, ...)，或直接原子驾驶"
}
```

- `reason.class` 复用 `classify_stuck` 的分类体系（reflect.py），保持一致语义。
- 宿主拿到 observation 可**直接决策下一步**，省掉重新截图枚举的一轮往返。
- 完成时返回 `{"status": "done", ...}`——协议统一，宿主只需处理两种返回。

## 5. MCP 工具面（终态）

| 工具 | 环 | 说明 | 依赖 |
|---|---|---|---|
| `get_capabilities` | — | §3 manifest | 无模型，立即可做 |
| `list_skills` / `run_skill` | 内环 | 宏查询/回放（适配判断+检查点+bailout） | experience-replay 实现 |
| `run_task` | 中环 | Jev 决策循环；Jev 未训练前 = 现状 VLM 循环 | desktop-jev |
| 原子工具 18 个（现状） | 外环 | 不变 | 已实现 |
| 全部执行类工具 | — | 返回协议统一为 done / EscalationResult | 协议改造 |

## 6. 实施阶段（协议先行，模型后到）

| 阶段 | 内容 | 依赖模型？ | 验收 |
|---|---|---|---|
| **A 协议层** | `get_capabilities` + EscalationResult 统一返回 + `run_task` escalate 路径（VLM 循环版） | ❌ 立即可做 | ✅ 已完成 2026-09-18：manifest（诚实标注 skill 未实现/引擎=vlm）+ 升级交接包（原因分类/进度/观测快照/建议），118 测试全绿 |
| **C 中环（优先）** | desktop-jev 按 TRAINING.md：mock 合成器 → SFT → 接入 `run_task`。owner 裁定：**Jev 是提速主力**（操作流程内化 + 并行单前向，对新任务成立），宏只覆盖重复任务降级为可选 | ✅ | 覆盖率 ≥70%、ECE ≤0.05、端到端 < 人类基线 |
| **B 内环（降级）** | experience-replay 实现：宏编译/hone/适配判断/退役 + `list_skills`/`run_skill` | ❌ | 重复任务 ≤3 次模型调用；宏漂移自动退役 |
| 持续 | 轨迹采集（已运转）：B 的宏原料 + C 的 SFT 原料 | — | 每周跑 `analyze_traces.py` 看覆盖率 |

北极星指标（每阶段对照）：**同一任务 agent 端到端耗时 vs 人类基线**——
A 之后宿主接管更快（省重观测），B 之后重复任务稳超人，C 之后新任务接近/超过人。

## 7. 与既有设计的关系

- experience-replay 的 bailout = 本文 §2.1 L2 + §4 在内环的实例化；宏 schema 直接沿用。
- desktop-jev 的升级协议 = §4 在中环的实例化；训练设计（TRAINING.md）不变。
- 本文新增的只有**对宿主的契约**：capabilities（§3）、EscalationResult（§4）、工具路由（§5）。
- 风险诚实清单沿用两文档；本文特有风险：**说明书与实现漂移**——manifest 是承诺，
  Jev 实际能力不及承诺会误导宿主。缓解：manifest 的 can/cannot 由阶段 C 实测数据回填，
  不从设计目标抄。
