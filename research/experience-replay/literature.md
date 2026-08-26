# 文献调研：GUI/桌面 Agent 的经验回放、VLA 低效问题与分层/技能库混合架构

> 调研时间：2026-08-27。面向 screen-use 的"经验回放（轨迹→参数化宏，语义锚点 + 检查点 + VLM 兜底）"设计提供文献支撑。
> 每条格式：[论文/工作名, 年份, 一句话方法, 关键数字/结论, URL]。找不到确切数字处如实标注。

## 1. GUI/Web Agent 的经验记忆与技能复用

- **Agent Workflow Memory (AWM)**, 2024 (CMU/MIT)。从成功轨迹中归纳"workflow"（抽象掉具体参数的公共子流程，文本或代码表示），存入 memory，测试时检索注入 prompt；支持 offline（从标注样例）与 online（从自生成且被验证为成功的预测）两种模式。WebArena 相对成功率 +51.1%（达 35.6% SOTA），Mind2Web 相对 step-wise SR +24.6%，且**成功任务所需 step 数减少**；cross-task/website/domain 泛化超出基线 8.9–14.0 绝对点，分布差距越大优势越明显。
  https://arxiv.org/abs/2409.07429
- **Synapse**, 2023/2024 (NTU)。trajectory-as-exemplar prompting + state abstraction + exemplar memory（embedding 相似度检索完整轨迹作为 few-shot 示例）。MiniWoB++ 上仅用 48 个任务的 demo 解决 64 个任务、平均 SR 99.2%；Mind2Web 上相对 MindAct 的 step SR +56%（GPT-3.5）、2.5×（CodeLlama-7B）。注意：记忆复用体现在 prompt 级，每步仍调 LLM。
  https://arxiv.org/abs/2306.07863
- **Agent S / S2 / S3**, 2024–2025 (Simular AI)。Agent S 引入 episodic memory（经验+教训）与 narrative memory；Agent S2 用 Mixture-of-Grounding（视觉/文本/结构三路 grounding 专家）+ Proactive Hierarchical Planning（Manager/Worker 两层）。Agent S OSWorld 20.6% → S2 34.5%（50-step，Claude-3.7），WindowsAgentArena 29.8%（+52.8% 相对）；S3（2025.10，bBoN wide scaling）OSWorld 69.9%→72.6%，超人类基线 72.36%。
  https://arxiv.org/abs/2504.00906 ；https://github.com/simular-ai/Agent-S
- **Learn-by-Interact**, 2025 (Google/HKU)。基于环境文档合成任务与交互轨迹，用"backward construction"（由子轨迹反推指令）对齐数据，再用 agentic retrieval（model-based + observation-based）检索示例做 ICL 或训练。ICL 最高 +12.2%（Claude-3.5），训练最高 +19.5%（Codestral-22B）；OSWorld 上 Claude-3.5 从 12.4%→22.5%。
  https://arxiv.org/abs/2501.10893
- **AutoGuide**, 2024 (LG AI Research 等)。从离线经验中的**对比轨迹对**（同任务成功/失败）提取 context-aware guideline（自然语言、条件化、标明适用上下文），测试时按当前上下文检索注入。在 ALFWorld/WebShop/WebArena 及多模态网站（GitHub/Google Flights/Coursera）上显著优于 ReAct/ExpeL（具体分数见论文表 1-3，本调研未逐项摘录）。
  https://arxiv.org/abs/2403.08978
- **AppAgent**, 2023/2024 (Tencent)。exploration 阶段自主探索或观摩 demo，为 UI 元素自动生成文档（knowledge base）；deployment 阶段逐步执行时按 RAG 查文档。AppAgent 基准：自动文档 SR 73.3% vs 无文档 GPT-4 基线 48.9%，自动文档效果≈人工文档。
  https://arxiv.org/html/2312.13771v3
- **AppAgentX**, 2025。结构化 memory chain（元素级记忆链）+ 进化机制：把常用动作序列升级为**高层动作**（macro），执行时跳过逐步推理。无记忆基线 SR 仅 16.9%；DroidTask 上 SR 46.3%→88.2%、平均任务耗时 106.2→56.29（单位：秒级/任务）、token 11.5k→5.1k；GPT-4o 平均每步耗时 43.5s→17.5s。是"宏回放替代逐步 LLM 推理"最直接的 GUI 先例与量化证据。
  https://arxiv.org/pdf/2503.02268v1

## 2. 技能库 / 宏方法

- **Voyager**, 2023 (NVIDIA/Caltech)。Minecraft lifelong agent：automatic curriculum 出题 → iterative prompting（环境反馈+执行错误+self-verification）生成可执行 JS 代码 → 成功代码按描述 embedding 存入 skill library，后续按语义相似度检索、组合。3.3× 独有物品、2.3× 行进距离、科技树里程碑快 15.3×；skill library 可迁移到新世界，赠予 AutoGPT 可提升其 zero-shot 泛化；消融：去掉 self-verification 掉 73%，去 curriculum 掉 93%。**关键机制：技能存的是代码（确定性执行），检索靠 embedding，入库前必须 self-verify。**
  https://arxiv.org/abs/2305.16291
- **SkillWeaver**, 2025 (OSU/CMU/Cisco)。把成功轨迹蒸馏为带参数、docstring 和 usage log 的 Python/Playwright API；三阶段：发现技能 → 练习 → hone（自动生成测试用例、执行、debug 修补），可让 agent "patch" 出错 API。WebArena 相对 SR +31.8%（25%→38%），真实网站 +39.8%；强 agent 合成的 API 迁移给弱 agent 最高 +54.3%。**与 screen-use 计划最接近的工作**：轨迹→参数化 API + 测试硬化，但仍在 Web/Playwright 域，且运行时仍由 LLM 调用 API。
  https://arxiv.org/abs/2504.07079
- **传统 RPA 录制回放脆弱性**（行业与学位论文证据，非单篇顶会）：RPA bot 依赖固定坐标/selector，UI 改版、布局漂移、弹窗、分辨率变化即失效；行业普遍估计年维护成本达初始建设成本的 30–50%（Forrester 常被引用），"RPA maintenance treadmill"。GUI 测试领域同样有 test fragility 文献（非功能性变更导致脚本断裂）。业界自愈合方案即"确定性 happy path + agent 恢复层"（Minicor 等）。
  https://minicor.com/blog/the-rpa-maintenance-problem-why-bots-break-and-how-to-fix-it ；https://orbilu.uni.lu/bitstream/10993/48254/1/rwemalika-thesis.pdf
- **WorkArena / WebArena**：前者（Drouin et al., 2024, arXiv:2403.07718）面向 ServiceNow 知识工作流；AWM 等工作都在 WebArena 上验证复用收益。WorkArena 细节本次未深入，未找到与经验回放直接相关的数字。

## 3. VLA 低效的证据（推理频率 / action chunking / 数据饥渴）

- **RT-2**, 2023 (Google DeepMind)。PaLI-X/PaLM-E 微调的 55B VLA，动作离散化为 token 自回归输出。推理仅约 1–3 Hz（多 TPU 云端），自回归逐 token 生成动作难以支撑高频控制。
  https://arxiv.org/abs/2307.15818 ；https://www.roboticscenter.ai/research/vla-models-comparison-2025
- **OpenVLA**, 2024 (Stanford)。7B 开源 VLA，在 Open X-Embodiment 970k episodes 上训练（64×A100×14 天，21,500 A100-hours）。RTX 4090 bf16 仅约 6 Hz、15GB 显存；4-bit 量化后 7.0GB 且 Bridge 任务 SR 不降（71.9% vs 71.3%）。训练/推理成本 vs 频率的量化锚点。
  https://arxiv.org/abs/2406.09246
- **OpenVLA-OFT**, 2025 (Stanford)。并行解码+连续动作头替代自回归 token，推理 26× 加速至 109 Hz——反向证明原 VLA 架构的频率瓶颈主要来自自回归 action token 生成。
  https://arxiv.org/abs/2502.19645
- **π0**, 2024 (Physical Intelligence)。3.3B（PaliGemma + 300M flow-matching action expert），action chunking（一次预测 H=50 个动作）支撑最高 50 Hz 灵巧控制；RTX 4090 上约 73ms/次推理；预训练用 10,000+ 小时灵巧操作数据（7 种机器人、68 任务）+ 全部 OXE——**数据饥渴的直接证据**。
  https://arxiv.org/abs/2410.24164
- **RTC: Real-Time Execution of Action Chunking Flow Policies**, 2025 (Physical Intelligence)。量化"VLA 慢 vs 控制快"矛盾：π0 3B 在 RTX 4090 上仅 KV-cache prefill 就 46ms，而目标 50Hz 控制周期只有 20ms；优化过的 7B OpenVLA 在 A100 上延迟仍 ≥321ms；有线 LAN 远程推理再加 13–21ms。同步推理在 chunk 间停顿会改变机器人动力学；temporal ensembling 在 +100/+200ms 注入延迟下直接失败（触发保护性急停），RTC（推理时 inpainting 做异步连续执行）显著更稳。**action chunking 之所以必要：模型推理延迟 ≫ 控制周期。**
  https://arxiv.org/html/2506.07339v1 ；https://www.pi.website/research/real_time_chunking
- **Open X-Embodiment (OXE)**, 2023。22 种机器人、1M+ 轨迹、500+ 技能的最大开源跨本体数据集；RT-X 证明跨本体训练能提升单机泛化，但 OOD 本体/任务仍需再微调（见 AWS VLA hub 对比表："out-of-distribution embodiments require fine-tuning regardless of model"）。数据规模与泛化的矛盾：百万级轨迹仍不足以免微调泛化。
  https://arxiv.org/abs/2310.08864 ；https://github.com/aws-samples/sample-vla-hub-on-aws

## 4. 分层 / 混合架构（快层-慢层）

- **SayCan**, 2022 (Google)。LLM 给"说什么"打分 × affordance value function 给"能不能做"打分，选技能序列由低层已学策略执行。101 个真实厨房任务：规划 SR 84%、执行 SR 74%；失败中 65% 源于 LLM、35% 源于 affordance——高层规划与低层技能各自是瓶颈。
  https://arxiv.org/abs/2204.01691
- **Code as Policies**, 2022/2023 (Google)。LLM 直接生成调用感知/控制 API 的 Python 策略代码；hierarchical code generation 可把新函数写进库中供复用（技能库雏形）。代码即策略 = 确定性回放 + 参数化的先例。
  https://arxiv.org/abs/2209.07753
- **Hi Robot**, 2025 (Physical Intelligence)。明确的 System 1/System 2 分层：高层 VLM（PaliGemma-3B 微调）低频运行，把开放指令/用户插话翻译成原子语言指令；低层 π0 VLA 高频执行 action chunk。高层在用户干预时立即重触发。用合成数据训练高层。对照实验含 "GPT-4o 高层 + π0 低层"（类 SayCan 升级版）与 flat VLA。
  https://arxiv.org/html/2502.19417v1 ；https://www.pi.website/research/hirobot
- **π0.5**, 2025 (Physical Intelligence)。单一模型两阶段推理：先自回归生成语言形式的"高层子任务"（chain-of-thought 式），再以 flow-matching 专家生成动作；面向开放世界泛化（多家庭环境移动操作）。高低层共享同一模型，区别在推理时解耦。注：开源权重只含低层 flow matching 部分。
  https://arxiv.org/abs/2504.16054 ；https://github.com/Physical-Intelligence/openpi
- **GR00T N1**, 2025 (NVIDIA)。双系统：System 2（Eagle VLM，~10 Hz）做语义推理，System 1（DiT flow matching，~120 Hz）做高频控制——VLA 内部已把"快慢层"固化到架构。
  https://github.com/aws-samples/sample-vla-hub-on-aws （二手来源；未找到官方逐字频率声明）
- **Options framework**, Sutton, Precup & Singh, 1999。option = ⟨initiation set, policy, termination condition⟩，基于 SMDP 的时间抽象；options 可被当原子动作做规划，且其 transition/reward model 可从经验学习。**screen-use 的"宏 + 前置条件 + 检查点终止/回退"正是 options 形式化的直接实例。**
  https://doi.org/10.1016/S0004-3702(99)00052-1 ；后续 Option-Critic: https://arxiv.org/abs/1609.05140

## 5. GUI grounding 与确定性执行

- **UGround**, 2024 (OSU)。"见 GUI 如人"的通用视觉 grounding 模型，7B，用 10M 合成 GUI 元素数据训练；ScreenSpot 平均 76.3（ScreenSpot-v2 平均 76.3），显著超 GPT-4o（18.8/20.1）。
  https://arxiv.org/abs/2410.05243
- **OS-Atlas**, 2024。GUI foundation action model（4B/7B），ScreenSpot-v2 平均 83.3（7B）；ScreenSpot-Pro（专业高分辨率桌面）上最好的原生模型也只有 18.9%——桌面专业软件 grounding 仍远未解决，佐证 screen-use 用 UIA 结构信息补视觉的价值。
  https://arxiv.org/abs/2410.23218 ；https://arxiv.org/html/2504.07981v1
- **ShowUI**, 2024 (NUS)。2B 轻量视觉-动作 GUI agent，UI-guided visual token selection 降低视觉 token 开销；ScreenSpot 平均 75.1。小模型专用化 grounding 足以替代大 VLM 做定位。
  https://arxiv.org/abs/2411.17465
- **ScreenSeekeR**（ScreenSpot-Pro 配套, 2025）：agentic 搜索框架（GPT-4o 规划 + 迭代裁剪 + OS-Atlas 定位），把 OS-Atlas-7B 在 ScreenSpot-Pro 从 18.9%→48.1%（+254% 相对），无需训练。"grounding 难时用规划循环换准确率"的对照组。
  https://arxiv.org/html/2504.07981v1
- **小结（本主题对"grounding 一次、回放多次"的支持）**：专用小模型（2–7B）grounding 精度已超 GPT-4o，说明定位能力可从大模型解耦——即 screen-use 的"语义锚点（UIA/小模型定位）+ 宏回放 + 失败回退大模型"在能力拆分上有依据；未找到明确"缓存 grounding 结果跨多次运行复用"的 GUI 论文（见 gaps.md）。
