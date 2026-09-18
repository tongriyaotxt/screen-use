# 变更记录 — 2026-09-18（desktop-jev 立项 + 阶段 0 轨迹采集）

## 新研究方向：research/desktop-jev/

从零训练类 Jev (TypeSafe AI, 2026-09-15) 的小型校准决策模型，**部分替代** VLM 逐步推理——
目的是加速，不承担文字输入（owner 裁定）。文档五件：

- `DESIGN.md`：定位（加速器非替代者）、UIA 元素表观测、收窄动作集、四头输出、
  升级 VLM 协议、三级 cascade（宏→小模型→VLM）、数据三条腿、5070Ti 预算
- `TRAINING.md`：训练设计（数据 schema、锚点特殊 token、四头损失、
  temperature scaling 校准、selective 曲线选 τ、消融计划）
- `literature.md`：Jev/RLCD 事实核实（厂商自报数字已标 ⚠️）、GUI agent 基线、
  grounding 小模型、评测套、校准方法
- `gaps.md`：桌面域校准步级决策无先例论证 + 6 条学术风险诚实清单
- `roadmap-kimi-2026-09-18.jpg`：路线图源文件（Kimi 对话截图）

## 阶段 0 已实现：结构化逐步轨迹采集

每次 `run_task` 自动攒 SFT 原料（TRAINING.md §1 schema）：

- **`memory.py` 新增 `TraceRecorder`**：每任务一个 `traces_full/{task_id}.jsonl`，
  每步落盘 `{goal, step_idx, foreground_title, elements, action, args, ok, detail, ts}`——
  与 `ExperienceStore.record_trace`（任务级摘要，供检索）并存，互不干扰。不存截图。
- **`agent.py::VisualLoop`** 挂采集钩子：act 每步 / done / think_error 三处落盘，
  写入失败静默（不影响任务结果，与 `_record` 同一原则）。
- **`config.py` 新增 `record_full_trace`**（默认 True，env `RECORD_FULL_TRACE`）。
- README Roadmap 挂上 desktop-jev 条目。

## 测试

111 passed（103 + 新增 8：TraceRecorder 往返/多任务文件/未 start_task 报错、
循环逐步落盘格式验证、开关关闭无产出、config 默认值）。

## 下一步

- 阶段 1：跑真实任务攒轨迹，统计无文本步占比与动作分布（覆盖率 kill criterion）
- 阶段 2：mock 应用合成器（tkinter 程序化窗口 + oracle 标签）

---

# 变更记录 — 2026-09-18 晚（首批轨迹采集 + 三个线上问题修复 + 阶段 1 首测）

## 首批真实轨迹（9 文件 105 步，`scripts/collect_traces.py`）

6 个常用任务（计算器/记事本/跨应用）全完成。采集中发现并修复三个线上问题：

1. **焦点抢占导致键盘输入落空**（主要失败模式）：MCP/脚本调用间隙终端抢回前台焦点。
   - `agent.py::_LOOP_PROMPT` 新增规则 5：键盘输入的连续操作（激活→输入→回车）必须用 batch 一次提交。
2. **重叠窗口点击穿透**：`_activate_window` 的 SetForegroundWindow 在焦点抢占防护下静默失效。
   - `tools.py`：激活后用 `GetForegroundWindow` 验证，失效则用 AttachThreadInput 附到前台线程重试。
3. **fail-safe 卡死连锁**：越界坐标被钳到屏幕边缘后，后续每次鼠标调用都抛 FailSafeException。
   - `executor.py` 新增 `_mouse_op`：捕获后移到屏幕中央复位并重试一次（用户主动急停仍会抛出）。
4. **UIA 配额截断丢失关键控件**（计算器 7/8/9 丢失事故根因）：多窗口时每窗配额 30，
   按 DFS 顺序截断且重复项吃配额。
   - `uia_tree.py`：每窗内先去重再按面积升序截取，小控件优先于大容器。

## 阶段 1 首次测量（`scripts/analyze_traces.py`）

- 直接可覆盖（点击/滚动/等待/完成）73.8%，含受限 type 乐观 90.5%，永升级 9.5%。
- **覆盖率假设方向成立**（DESIGN §10 阶段 1 已回填）；样本量小、任务类型单一，需持续采集。

## 测试

114 passed（111 + 新增 3：fail-safe 恢复/二次急停放行、配额前去重）。

---

# 变更记录 — 2026-09-18 深夜（cascade protocol 阶段 A：三环协议落地）

owner 三点裁定（内环适配判断+实时更新、Jev 说明书、失败退回宿主完全控制）
→ `research/cascade-protocol/README.md` 三环契约设计。阶段 A（协议层，不依赖模型）已实现：

- **`capabilities.py`**：`get_capabilities()` 三环说明书（skill/task/atomic 的 can/cannot、
  goal 写法、路由建议、升级协议）。诚实标注：skill 未实现、task 当前引擎 vlm、
  can/cannot 待 Jev 实测回填。
- **`tools.py::run_task`** 统一返回协议：完成 → `{"status": "done"}`；
  未完成 → **EscalationResult（带地图退）**：reason 分类（think_error/stuck/max_steps_reached）
  + 已做步骤 + 剩余目标 + **当前观测快照（元素表）** + 建议——宿主拿到即可直接接管，
  省掉重新截图枚举的一轮往返。
- **MCP 新增 `get_capabilities` 工具**（19 个），`run_task` 工具描述写入路由提示。
- 顺手修复：`mcp_server.py` 过时的命令入口注释（rpa-mcp-server → screen-use-mcp）。

## 测试

118 passed（114 + 新增 4：manifest 结构/诚实标注、done 返回、escalation 交接包含观测快照、
think_error 分类）。

---

# 变更记录 — 2026-09-18 深夜②（owner 裁定 Jev 优先 + 腿 2 mock 合成器落地）

## 优先级调整

owner 裁定：**Jev（中环）是提速主力**——操作流程内化 + 并行单前向，对新任务成立；
宏（内环）只覆盖重复任务，降级为可选。路线调整为 A ✅ → C（Jev）→ B（宏，可选）。

## 腿 2：mock 应用合成器（scripts/mock_env.py）

桌面版 MiniWoB++：程序化窗口 + 模板任务 + oracle 标签，零模型成本量产 SFT gold 数据。

- **技术选型实测**：tkinter UIA 不可用（按钮无名、输入框/列表不可见）→ PySide6
  （accessibleName → EditControl 实名、ListItemControl 齐全）。pyside6 装入 .venv
  （仅合成器依赖，screen_use 包不背 Qt）。
- **6 类任务模板**：click_button / type_field / form_submit / list_select / scroll_list /
  blocked_missing（BLOCKED 类首批样本，真实数据该类为 0）。
- **干扰项内建**：每窗 2-4 输入框 + 3-6 按钮随机布局，目标只占其一——指针头必须学会选择。
- **train/serve 一致性**：观测走 screen_use 真实 UIA 管线（过滤到 mock 窗口 + id 重排；
  原始全局 id 仅用于执行，不落入记录）。
- 修了两个实现 bug：局部/全局 id 错位导致执行打错元素；Qt 列表虚拟化 → 滚动后末项
  才进 UIA（滚轮对 QListWidget 不生效，用 Qt 滚动条保证状态，记录动作仍为 scroll——
  合成器专属问题，真实应用无此问题）。
- 冒烟 17/17 oracle 执行成功；量产 500 episodes（~2K 步）进行中 → `~/.screen_use/traces_mock/`。

---

# 变更记录 — 2026-09-19 凌晨（mock v2 范式扩展 + SFT v0 训练完成）

## mock v2：零干扰 + 15 类交互范式（owner 反馈驱动）

- **零干扰改造**：UIA InvokePattern/SetFocus/ExpandCollapse 点击（不动鼠标）、
  WA_ShowWithoutActivating（不抢焦点）、Qt 滚动条（不滚轮）。
- **grabMouse 事故**：下拉/菜单弹层会抓取用户鼠标 → 弹层生命周期管理
  （只有打开步之后允许弹层存活到下一步观测，其余立即强制关闭）。
- **窗口出屏 bug**：Qt 用逻辑像素（1280×800）却给了物理坐标 → 修正到屏幕内右下角。
- **范式扩展**：语义对抗（确定/取消、保存/另存为近名干扰）、状态依赖（选中才能删、
  必填才能提交）、搜索流、向导流、菜单/树/表格/模态弹窗、下拉、勾选/单选、
  BLOCKED×2（不存在/禁用）——54/54 oracle 执行成功。
- **screen_use 本体修复**：MAX_WINDOWS 5→8（真实桌面 13 顶层窗口会整窗丢失目标）、
  前台窗口配额保底 50%。

## SFT v0 训练完成（阶段 C 里程碑）

- 数据：3,100 步（mock v1+v2 gold + 真实轨迹 silver），按 episode 划分留出 330 步。
- 管线 debug：span 标签 -1 混入 CE（loss=inf）、batched no-element 槽列错位（loss=inf 根因）、
  calibrate 误从基座加载 backbone（丢弃训练权重）。
- **结果（e3，Qwen3-0.6B，5070Ti 27min）**：留出集 op_acc 94.5% / joint_acc 68.8%；
  校准后 ECE 0.0（等张 in-sample 乐观）；τ=0.5 → 覆盖率 70.9%、快路径风险 14.1%。
- **真实世界冒烟（诚实）**：真实桌面观测（61 元素，分布外）→ 模型主动低置信升级，
  不乱点——升级协议按设计工作；同时暴露 R4 校准漂移：mock→real 泛化是下一步主战场。
- 延迟 p50 229ms（目标 <100ms，优化空间：量化/更短上下文）。
- serve.py 就绪：观测进 → {action, element_id, confidence, escalate} 出，协议同阶段 A。
