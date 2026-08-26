# 文献空白分析（gaps）

> 配合 literature.md 阅读。回答三个问题：现有工作没解决什么？screen-use 的方案是否有先例？具身迁移路径上文献支持什么、缺什么？

## 1. 现有文献没解决的问题

**a) 经验复用停留在"prompt 级"，没有做到"执行级"。**
AWM、Synapse、AutoGuide、AppAgent、Learn-by-Interact 的共同模式是：把经验（workflow/轨迹/guideline/文档）检索进 prompt，**每一步仍要调大模型做决策**。它们提升的是成功率和泛化，而不是消灭 LLM 调用。AWM 报告"减少 step 数"但每步推理成本不变；Synapse 自己也承认 "high inference latency is a major concern"。**"把已验证的轨迹变成不依赖 LLM 的确定性执行体"这一层抽象，GUI agent 文献里基本缺位**——最接近的是 AppAgentX 的 evolved high-level actions（每步 43.5s→17.5s）和 SkillWeaver 的 API，但前者仍是 LLM 驱动的高层动作调度，后者运行时仍由 LLM 选择/填参调用 API，且都在 Web/Mobile 域。

**b) 桌面（Windows/桌面应用）域完全空白。**
所有经验记忆工作都在 Web（WebArena/Mind2Web）或 Android。Windows 桌面依赖 UIA/accessibility tree 而非 DOM，grounding 文献（ScreenSpot-Pro）反而证明纯视觉 grounding 在专业桌面软件上只有 ~19–48% 准确率——UIA 结构信息是桌面域独有的杠杆，没有任何经验回放工作利用它。

**c) 回放的可靠性机制（验证/回退）没人系统化。**
SkillWeaver 有"API hone"（离线测试硬化）和"patch on failure"，但这是离线阶段的事；运行时的"检查点验证 + 失败时回退慢速推理"这一组合，GUI 文献中没有对应物。传统 RPA 行业的 self-healing（确定性脚本 + agent 恢复层，如 Minicor）有工程实践但没有学术评估——这是一个"工程界在做、学术界没量化"的空白。

**d) 没有人回答"什么时候该回放、什么时候该探索"。**
Options framework 给出形式化（initiation set / termination），但 LLM agent 时代没人把"宏的可适用性判定、漂移检测、失效退役"做成学习问题。AWM 课件里自己都留了 open question：workflow 被污染/过时怎么办。

## 2. screen-use 方案的先例分析

方案三件套：**语义锚点定位 + 检查点验证 + 失败回退 VLM 循环**。

- **语义锚点**：部分先例。Synapse 的 state abstraction、AppAgent 的元素文档是"语义化状态"的远亲；grounding 小模型（UGround/OS-Atlas）证明定位可从大模型解耦。但"用 UIA 无障碍树做锚点并在回放时重新解析定位"未见先例——这是桌面域独有的增量。
- **检查点验证**：SkillWeaver 的 reward model / usage log、Voyager 的 self-verification 是先例，但都用于"入库前"而非"运行时"。RPA self-healing 的 reflection step（每步验证屏幕状态再继续）是工程先例。
- **失败回退 VLM**：最接近 Hi Robot 的高层重触发、Agent S2 的 adaptive navigation（失败后换 grounding 专家/路径），但"宏执行中途失败→把控制权交还 VLM 循环续跑"的明确机制未见发表。
- **结论：组合层面没有先例。** 单独组件各有远亲，但"轨迹→参数化宏（语义锚点重定位）→运行时检查点→回退 VLM"这一端到端闭环，在 GUI agent 学术文献中是新的，且恰好补上 §1a/§1b/§1c 三个空白。它本质是把 Voyager 的"代码技能库"思想（技能=可确定性执行的代码）从 Minecraft 搬到 Windows GUI，用 UIA 锚点替代 API 调用解决鲁棒性。

## 3. 具身智能迁移路径：文献支持什么、缺什么

**支持（可以引用的论据链）：**
1. "单纯 VLA 跟不上控制频率"有硬数据：RT-2 1–3Hz、OpenVLA 6Hz、π0 在 RTX 4090 上 prefill 46ms vs 50Hz 周期的 20ms 预算（RTC 论文）。"大模型慢→必须降频调用"是社区共识，且催生了 action chunking、OpenVLA-OFT 并行解码、GR00T N1 双系统等一整类工作。
2. 快慢分层是具身智能的主流解：Hi Robot（VLM 高层 + VLA 低层）、π0.5（同模型两阶段）、GR00T N1（10Hz System2 / 120Hz System1）。Kahneman System 1/2 类比已被 Hi Robot 明确引用。
3. "技能即代码"已被验证有效：Voyager 技能库（3.3×/15.3× 数据 + 消融证明 self-verification 关键）、Code as Policies、SkillWeaver。技能库的跨 agent/跨环境迁移也有证据（SkillWeaver +54.3%）。
4. Options framework 提供形式化基础：宏=option，语义锚点≈initiation set，检查点≈termination condition，回退 VLM≈policy over options 重选择。

**缺失（owner 的混合架构主张超出文献的地方）：**
1. **现有快慢分层的"快层"全是 learned policy（神经网络），没有一个是"经验回放的确定性宏"。** owner 设想的"技能回放快层"更接近工业运动规划/RPA 而非 VLA 社区的 System 1——这在文献里没有对应实验，是主张而非结论。诚实的说法：文献支持"分层"，不支持"快层必须是回放"——π0 这类工作会主张快层也应该是 learning policy（因为物理世界连续、需要闭环纠偏）。
2. **回放与学习的接口问题无人研究**：何时把 VLA 探索出的行为蒸馏成宏？宏的漂移如何反向触发再学习？GUI 域可假设"重复任务状态可枚举"，物理世界不行——这条迁移路径上文献没有桥。
3. **数据饥渴的证据都在"预训练规模"层面**（OXE 1M 轨迹仍不够、π0 10k 小时），没有工作量化"回放层能替代多少 VLA 数据需求"——screen-use 若能量化"回放覆盖率 vs VLM 调用次数"，本身就是填补空白的贡献。
4. RTC 论文其实给了最有力的"分层/异步"论据：推理延迟只会随模型变大而增加（+100/+200ms 注入实验），异步执行不是权宜之计而是必然架构——这一点对 owner 的主张是加分项。

## 4. 给设计文档的一句话定位建议

把 screen-use 的经验回放定位为：**"Voyager-style skill library 在 Windows GUI 域的实现 + options framework 的形式化 + RPA self-healing 的工程验证补全"**，并把"GUI 回放宏 = 具身混合架构快层的离散可枚举特例"作为向 VLA 迁移的论证起点，同时明确承认 §3 缺失 1 的学术风险（快层是否该 learning 是开放问题）。
