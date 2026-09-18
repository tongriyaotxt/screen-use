# 文献调研（desktop-jev）

> 2026-09-18。每条给关键数字与出处。标注 ⚠️ 的为厂商自报/未经独立验证。
> 回答一个问题：**"小型校准决策模型加速 GUI agent 循环"这条路线，文献支持到哪一步？**

## 1. System One 决策模型：Jev 与 RLCD（本设计的范式来源）

**TypeSafe AI / Jev**（2026-09-15 发布，early access API）。创始人 Diogo Almeida
（OpenAI 前研究员，InstructGPT/RLHF 共同作者）。

- 架构：非自回归**并行采样**；输出空间预先枚举（Choice ≤255 选项 / Score / Noul 三原语），
  schema 符合性是数学保证而非采样结果——"can't hallucinate" 的真实含义是"不能越出 schema"，
  不等于不能选错（CEO 在 HN 亲自承认）。
- 训练：**RLCD（Reinforcement Learning for Calibrated Decisions）**——优化校准
  （说 70% 把握则 70% 对），区别于 RLHF（人类偏好）与 RLVR（可验证奖励）。
- 数字（⚠️ 全部厂商自报）：70–500ms 端到端（对照 frontier LLM 3–329s）；
  $0.042/MTok 输入、输出免费；自家 workflow evals 报 193.6× 快、444.6× 便宜，
  参照系是 GPT-6 Astra / Fable 5.1 的**平均预测**而非 ground truth（HN 已质疑）；
  上下文 32K；**无图像输入**。无架构论文、无开源权重。
- 对我们的意义：证明"放弃字符串生成 → 单次前向出校准决策"是成立的范式；
  但 Jev 是纯文本 API，**不提供桌面 agent 可用的工件**，且闭源——自训是唯一路径。
- 来源：https://typesafe.ai/blog/introducing-system-one-models-and-jev ；
  HN 讨论（2026-09-15，1500+ 票）；
  https://www.developersdigest.tech/blog/typesafe-jev-system-one-models-release-guide-2026

**Kahneman System 1/2**：《Thinking, Fast and Slow》(2011)。Jev 命名即源于此。
GUI/机器人域的快慢分层先例见 experience-replay/literature.md（Hi Robot、GR00T N1 等）。

## 2. GUI agent 基线：现状全是"每步一次大模型"

**UI-TARS**（ByteDance/Tsinghua，2025-01，arXiv 2501.12326）。原生 GUI agent，
screenshot→action 端到端。UI-TARS-1.5 报 OSWorld 42.5%（⚠️ 自报）。
其 72B/7B 开源权重证明中小模型可以学 GUI 决策，但它是自回归 VLM，
每步仍是完整生成——没有步级快路径概念。

**Aguvis**（2024-12，arXiv 2412.04454）。统一纯视觉 GUI agent 框架（image+action
两阶段），跨 web/桌面。同样每步一次大模型前向。

**ShowUI**（2024-11，arXiv 2411.17465；ShowUI-2B）。UI 视觉 token 稀疏化降 75%
推理成本——压缩的是"看"的成本，决策仍是自回归 LLM。与本设计正交，可叠加。

**OS-Atlas / UGround**（OS-Atlas: arXiv 2410.23218；UGround: arXiv 2501.05461）。
**grounding 小模型**（2B–7B）把"指哪"从大模型解耦：给文本描述→输出坐标。
OS-Atlas-7B 在 ScreenSpot 等 grounding benchmark 接近 GPT-4o 水平。
**与本设计的关键边界**：它们只做单元素定位，不做步级决策（点/滚/完成/认输），
也没有校准置信度——但它们证明 2B 量级足以承载 GUI 定位能力，是 §5.2 规模选型的证据。

**Agent S / S2**（arXiv 2410.08164 / 2504.00906）。经验增强 + agentic workflow，
OSWorld SOTA 梯队（S2 报 34.5% @15 步，⚠️ 自报）。其 adaptive navigation 思想
（失败换专家）与升级协议同源，但换的是"另一个大模型"，不是"更小的快模型"。

**click/scroll 步占比证据**：Mind2Web（arXiv 2306.06070）任务轨迹统计中
click/type/select 为主、scroll 次之；WebArena 轨迹分析（arXiv 2307.13854）显示
click+scroll 占动作总数 >60%。桌面域无公开等价统计——**这正是阶段 1 要测的**
（DESIGN §1 覆盖率假设）。

## 3. 环境 / 评测套

**MiniWoB / MiniWoB++**（Shi et al. 2017, ICML, arXiv 1703.09725；Liu et al. 2018,
NeurIPS, arXiv 1802.08802）。~100 个程序化网页小任务，oracle 奖励自动生成——
**"合成环境给免费完美标签"的鼻祖**，本设计腿 2 的直接模板。仅 Web。

**WebShop**（Yao et al. 2022, NeurIPS, arXiv 2207.01206）。118 万商品的模拟电商，
1.2K 指令；rule-based agent ~29% vs 人类 ~59%。证明模拟环境可迁移部分能力到真实站点。

**WebArena**（Zhou et al. 2023, ICLR 2024, arXiv 2307.13854）。812 任务自托管站点；
GPT-4 文本 agent 14.4% vs 人类 78.2%。

**OSWorld**（Xie et al. 2024, NeurIPS 2024 D&B, arXiv 2404.07972）。369 真实桌面任务
（Ubuntu VM，程序化 verifier）；发布时最佳 agent 12.2% vs 人类 72.4%。
桌面评测的事实标准。

**WindowsAgentArena**（Microsoft，2024-09，arXiv 2409.08264）。150+ Windows 任务，
**唯一 Windows 原生桌面 benchmark**；其 Navi agent 报 19.5% vs 人类 74.7%。
与本设计同域（Windows + UIA 可得），子集可直接用作外部参照。

**ScreenSpot / ScreenSpot-Pro**（Cheng et al. 2024, arXiv 2404.07972 系；
Pro: arXiv 2504.07981）。GUI grounding 评测；Pro 版（专业软件高分辨率）上
text-supervised grounding 专家仅 ~18.9%，最好模型 ~48%——**纯视觉在桌面专业
场景的硬上限证据**，支撑"UIA 结构化观测是桌面域杠杆"。

## 4. 校准（RLCD 的平价替代）

**Temperature Scaling / 现代网络校准**（Guo et al. 2017, ICML, arXiv 1706.04599）。
深度网络普遍过自信；单参数温度缩放即可显著降 ECE——SFT 后校准的标准第一步，
本设计 RLCD 的零成本近似。

**等张回归 / Platt scaling**：scikit-learn 成熟实现，留出集拟合，无额外训练。

**选择性预测 / learning to defer**（Geifman & El-Yaniv 2017, arXiv 1705.08500；
Madras et al. 2018, arXiv 1803.01783）。"置信度低则交权"的理论基础：
给定风险预算求最优覆盖率——升级协议 τ 的调参可直接引用其 selective risk 框架。

## 5. 教师蒸馏与省钱采样

**Committee / 分歧采样省钱法**：多个本地量化教师一致→免费标签，分歧→强 API 仲裁。
经典主动学习（Query by Committee, Seung et al. 1992）在 LLM 蒸馏时代的常规做法。

**5070Ti 本地教师可行性**：Qwen3-8B AWQ/GPTQ 约 6GB，14B 约 10GB，16GB 显存均可跑；
Qwen3 系列 Apache-2.0（2025-04 发布，0.6B/1.7B/4B/8B/14B/32B/235B）。
基座选型 Qwen3-0.6B/1.7B 同宗，蒸馏分布更接近。

**Distilling Step-by-Step**（Hsieh et al. 2023, ACL, arXiv 2305.02301）：
教师 rationale + 标签联合蒸馏，小模型用更少数据追平教师——腿 3 的方法论参照。

## 6. 与本设计的三层关系总结

| 层 | 直接先例 | 空白（详见 gaps.md） |
|---|---|---|
| 并行类型化输出 + 校准目标 | Jev/RLCD（文本域，闭源） | GUI 决策域无人做 |
| 小模型 GUI 定位 | OS-Atlas/UGround（grounding 头） | 步级决策（非定位）无人小模型化 |
| 置信度交权 | 选择性预测理论；Agent S2 换专家 | "换小模型→大模型"的校准 cascade 无人系统化 |
